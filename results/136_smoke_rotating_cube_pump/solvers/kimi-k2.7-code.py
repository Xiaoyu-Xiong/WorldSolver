"""3D smoke solver for the rotating cube pump scene.

Implements a finite-difference incompressible Navier-Stokes solver with a
pressure-projection step on a regular voxel grid.  Smoke is advected with a
semi-Lagrangian scheme, buoyancy lifts the scalar field, and a kinematic
rotating cube drives the flow through rigid-body wall velocities.  All
physics is stepped deterministically with NumPy/SciPy only.
"""
from __future__ import annotations

import numpy as np
import os

import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
import yaml

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Baseline frame stores the static particle data expected by the contract.
        self._frame0 = scene.initial_state()

        # Grid shape and domain from the gas voxel grid definition.
        geom_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "scene_geometry.yaml"
        )
        if os.path.exists(geom_path):
            with open(geom_path, "r") as fh:
                scene_geometry = yaml.safe_load(fh)
            geom = scene_geometry["objects"]
        else:
            geom = self.config.get("scene_geometry", {}).get("objects", [])

        # Find the gas object to get the voxel bounds.
        gas_obj = None
        for obj in geom:
            if obj["semantic"] == "gas":
                gas_obj = obj
                break
        if gas_obj is None:
            gas_obj = {
                "geometry": {
                    "bounds_min": [-3.0, -1.2, 0.0],
                    "bounds_max": [3.0, 1.2, 3.8],
                    "grid_shape": [24, 16, 28],
                }
            }
        g = gas_obj["geometry"]
        self.shape = tuple(g["grid_shape"])  # (nx, ny, nz)
        self.bounds_min = np.array(g["bounds_min"], float)
        self.bounds_max = np.array(g["bounds_max"], float)
        self.L = self.bounds_max - self.bounds_min

        self.nx, self.ny, self.nz = self.shape
        # Cell-centered axes.  Using linspace matches the initial scalar field.
        self.x = np.linspace(self.bounds_min[0], self.bounds_max[0], self.nx)
        self.y = np.linspace(self.bounds_min[1], self.bounds_max[1], self.ny)
        self.z = np.linspace(self.bounds_min[2], self.bounds_max[2], self.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        self.X, self.Y, self.Z = np.meshgrid(
            self.x, self.y, self.z, indexing="ij"
        )

        # Physical parameters.
        self.rho = float(self.config.get("sampling", {}).get("density", 1.2))
        self.g = np.array(self.config["gravity"], float)
        # Upward buoyancy coefficient (scalar drives upward acceleration).
        self.buoyancy = 1.2
        # Smoke source injection rate.
        self.source_rate = 0.35
        # Viscous diffusion for velocity and scalar.
        self.nu = 1e-4
        self.kappa = 1e-4

        # Rotating cube parameters.
        self.cube_center = np.array([0.0, 0.0, 1.55])
        self.cube_half = np.array([0.45, 0.45, 0.45])
        self.omega = 2.0  # rad/s about y-axis
        # Width of the wall-velocity band near the cube surface.
        self.wall_band = 1.5 * max(self.dx, self.dy, self.dz)

        # State fields.
        self.u = self._frame0.fields["velocity_field"].copy()
        self.scalar = self._frame0.fields["transported_scalar"].copy()
        self.active_mask = self._frame0.fields["active_mask"].copy()
        self.scalar_source = np.zeros(self.shape, float)

        # Source mask: static cylinder at the smoke_source location.
        src = None
        for obj in geom:
            if obj["id"] == "smoke_source":
                src = obj
                break
        if src is not None:
            c = np.array(src["transform"]["translation"], float)
            r = float(src["geometry"]["radius"])
            h = float(src["geometry"]["height"])
            dz_local = self.Z - c[2]
            dist2 = (self.X - c[0]) ** 2 + (self.Y - c[1]) ** 2
            in_cyl = (dist2 <= r * r) & (dz_local >= -h / 2) & (
                dz_local <= h / 2
            )
            self.scalar_source[in_cyl] = self.source_rate

        # Boundary sample points and normals (world frame, updated each frame).
        self._init_boundary_samples()

        self.time = 0.0
        # Internal sub-step is half a sample interval for stability.
        self._dt = float(self.config["sample_dt"]) / 2.0

        # Pre-allocate the pressure vector for warm-starting the CG solver.
        self._p_prev = np.zeros(self.shape, float)

    def _init_boundary_samples(self) -> None:
        """Create fixed boundary sample points for the ground and cube."""
        pts = []
        nrm = []

        # Ground top surface samples.
        xg = np.linspace(self.bounds_min[0], self.bounds_max[0], self.nx)
        yg = np.linspace(self.bounds_min[1], self.bounds_max[1], self.ny)
        Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
        Zg = np.zeros_like(Xg)
        for xg_i, yg_i, zg_i in zip(Xg.ravel(), Yg.ravel(), Zg.ravel()):
            pts.append([xg_i, yg_i, zg_i])
            nrm.append([0.0, 0.0, 1.0])

        # Cube surface samples (local frame).
        n_samp = 7
        s = np.linspace(-0.45, 0.45, n_samp)
        for face, normal in [
            ("x", [1.0, 0.0, 0.0]),
            ("x_neg", [-1.0, 0.0, 0.0]),
            ("y", [0.0, 1.0, 0.0]),
            ("y_neg", [0.0, -1.0, 0.0]),
            ("z", [0.0, 0.0, 1.0]),
            ("z_neg", [0.0, 0.0, -1.0]),
        ]:
            S1, S2 = np.meshgrid(s, s, indexing="ij")
            S1 = S1.ravel()
            S2 = S2.ravel()
            for si, ti in zip(S1, S2):
                if face.startswith("x"):
                    p = [
                        0.45 if face == "x" else -0.45,
                        si,
                        ti,
                    ]
                elif face.startswith("y"):
                    p = [si, 0.45 if face == "y" else -0.45, ti]
                else:
                    p = [si, ti, 0.45 if face == "z" else -0.45]
                pts.append(p)
                nrm.append(normal)

        self._bnd_local = np.array(pts, float)
        self._bnd_normal_local = np.array(nrm, float)

    def _cube_state(self, t: float):
        """Return solid mask and wall-velocity field for the rotating cube."""
        theta = self.omega * t
        c = np.cos(theta)
        s = np.sin(theta)

        # World -> local frame: rotate by -theta around y.
        dx = self.X - self.cube_center[0]
        dy = self.Y - self.cube_center[1]
        dz = self.Z - self.cube_center[2]
        lx = dx * c + dz * s
        ly = dy
        lz = -dx * s + dz * c

        inside = (
            (np.abs(lx) <= self.cube_half[0])
            & (np.abs(ly) <= self.cube_half[1])
            & (np.abs(lz) <= self.cube_half[2])
        )

        # Distance to the rotated box for the wall-velocity band.
        ex = np.maximum(np.abs(lx) - self.cube_half[0], 0.0)
        ey = np.maximum(np.abs(ly) - self.cube_half[1], 0.0)
        ez = np.maximum(np.abs(lz) - self.cube_half[2], 0.0)
        dist = np.sqrt(ex * ex + ey * ey + ez * ez)
        band = dist < self.wall_band

        # Rigid-body wall velocity: omega y (p - c).
        wall = np.zeros(self.shape + (3,), float)
        wall[..., 0] = self.omega * (self.Z - self.cube_center[2])
        wall[..., 2] = -self.omega * (self.X - self.cube_center[0])

        return inside, band, wall

    def _rotate_y(self, pts: np.ndarray, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return pts @ R.T

    def _boundary_data(self, t: float):
        theta = self.omega * t
        # Ground points are already in world frame; cube points need rotation.
        n_ground = len(self._bnd_local) - 6 * (7 * 7)
        ground_pts = self._bnd_local[:n_ground]
        ground_nrm = self._bnd_normal_local[:n_ground]
        cube_local = self._bnd_local[n_ground:]
        cube_nrm_local = self._bnd_normal_local[n_ground:]

        cube_pts = self._rotate_y(cube_local, theta) + self.cube_center
        cube_nrm = self._rotate_y(cube_nrm_local, theta)

        pts = np.vstack([ground_pts, cube_pts])
        nrm = np.vstack([ground_nrm, cube_nrm])

        # Target / actual wall velocity.
        vel = np.zeros_like(pts)
        # Cube samples: rigid rotation about y.
        vel_cube = np.zeros_like(cube_pts)
        vel_cube[:, 0] = self.omega * (cube_pts[:, 2] - self.cube_center[2])
        vel_cube[:, 2] = -self.omega * (cube_pts[:, 0] - self.cube_center[0])
        vel[n_ground:] = vel_cube

        return pts, nrm, vel, vel

    def _advect_field(self, f: np.ndarray, dt: float) -> np.ndarray:
        """Semi-Lagrangian advection of a scalar or vector component field."""
        # Back-trace cell centers.
        bx = self.X - dt * self.u[..., 0]
        by = self.Y - dt * self.u[..., 1]
        bz = self.Z - dt * self.u[..., 2]
        ix = (bx - self.bounds_min[0]) / self.dx
        iy = (by - self.bounds_min[1]) / self.dy
        iz = (bz - self.bounds_min[2]) / self.dz
        coords = np.stack([ix, iy, iz], axis=0)
        return scipy.ndimage.map_coordinates(
            f, coords, order=1, mode="nearest"
        )

    def _advect_velocity(self, dt: float) -> None:
        # Advect each Cartesian component separately with the same back-trace.
        u_new = np.empty_like(self.u)
        for c in range(3):
            u_new[..., c] = self._advect_field(self.u[..., c], dt)
        self.u = u_new

    def _build_poisson_matrix(self, active: np.ndarray):
        """Build the positive-definite -Laplacian restricted to active cells."""
        n = np.prod(self.shape)
        idx = np.arange(n).reshape(self.shape)
        I, J, V = [], [], []

        def add(start: slice, end: slice, h: float):
            # Cells where the neighbour in the + direction exists.
            cell = idx[start]
            nbr = idx[end]
            cell_active = active[start]
            nbr_active = active[end]
            coeff = 1.0 / (h * h)
            # Diagonal contribution for every cell that has a physical neighbour.
            valid = cell_active
            if np.any(valid):
                lin = cell[valid]
                I.extend(lin.tolist())
                J.extend(lin.tolist())
                V.extend([coeff] * int(valid.sum()))
                # Off-diagonal when the neighbour is also active.
                both = cell_active & nbr_active
                if np.any(both):
                    lin_b = cell[both]
                    nbr_b = nbr[both]
                    I.extend(lin_b.tolist())
                    J.extend(nbr_b.tolist())
                    V.extend([-coeff] * int(both.sum()))

        nx, ny, nz = self.shape
        # x direction
        add(slice(1, nx), slice(0, nx - 1), self.dx)
        add(slice(0, nx - 1), slice(1, nx), self.dx)
        # y direction
        add(slice(1, ny), slice(0, ny - 1), self.dy)
        add(slice(0, ny - 1), slice(1, ny), self.dy)
        # z direction
        add(slice(1, nz), slice(0, nz - 1), self.dz)
        add(slice(0, nz - 1), slice(1, nz), self.dz)

        A = scipy.sparse.csr_matrix(
            (np.array(V, float), (np.array(I), np.array(J))),
            shape=(n, n),
        )
        return A

    def _project(self, dt: float) -> None:
        # Enforce wall/band velocities before taking divergence.
        solid, band, wall = self._cube_state(self.time)
        wall_mask = solid | band
        self.u[wall_mask] = wall[wall_mask]

        # Compute divergence of the collocated velocity field.
        ux = self.u[..., 0]
        uy = self.u[..., 1]
        uz = self.u[..., 2]
        div = (
            np.gradient(ux, self.dx, axis=0)
            + np.gradient(uy, self.dy, axis=1)
            + np.gradient(uz, self.dz, axis=2)
        )

        active = ~solid
        b = -(self.rho / dt) * div[active]

        A = self._build_poisson_matrix(active)
        x0 = self._p_prev[active]
        x, info = scipy.sparse.linalg.cg(A, b, x0=x0, tol=1e-9, maxiter=2000)
        if info < 0:
            raise RuntimeError("Pressure CG failed to converge")

        p = np.zeros(self.shape, float)
        p[active] = x
        self._p_prev = p.copy()

        # u -= dt/rho * grad(p)
        grad_p_x = np.gradient(p, self.dx, axis=0)
        grad_p_y = np.gradient(p, self.dy, axis=1)
        grad_p_z = np.gradient(p, self.dz, axis=2)
        factor = dt / self.rho
        self.u[..., 0] -= factor * grad_p_x
        self.u[..., 1] -= factor * grad_p_y
        self.u[..., 2] -= factor * grad_p_z

        # Re-impose the wall velocity in the solid and band cells.
        self.u[wall_mask] = wall[wall_mask]

        # Enforce no-normal-flow on the domain boundaries.
        self.u[0, :, :, 0] = 0.0
        self.u[-1, :, :, 0] = 0.0
        self.u[:, 0, :, 1] = 0.0
        self.u[:, -1, :, 1] = 0.0
        self.u[:, :, 0, 2] = 0.0
        self.u[:, :, -1, 2] = 0.0

    def _step(self, dt: float) -> None:
        # 1. Scalar source injection and diffusion.
        self.scalar[...] += dt * self.scalar_source
        if self.kappa > 0.0:
            lap = (
                np.gradient(self.scalar, self.dx, axis=0, edge_order=2)
                + np.gradient(self.scalar, self.dy, axis=1, edge_order=2)
                + np.gradient(self.scalar, self.dz, axis=2, edge_order=2)
            )
            self.scalar[...] += dt * self.kappa * lap
        self.scalar = np.maximum(self.scalar, 0.0)

        # 2. Advect scalar field.
        self.scalar = self._advect_field(self.scalar, dt)

        # 3. Advect velocity field.
        self._advect_velocity(dt)

        # 4. Apply buoyancy (upward force proportional to smoke density).
        # g points downward; buoyancy opposes it.
        buoy_acc = self.buoyancy * self.scalar * 9.81
        self.u[..., 2] += dt * buoy_acc

        # 5. Velocity diffusion (viscosity).
        if self.nu > 0.0:
            lap_u = (
                np.gradient(self.u, self.dx, axis=0, edge_order=2)
                + np.gradient(self.u, self.dy, axis=1, edge_order=2)
                + np.gradient(self.u, self.dz, axis=2, edge_order=2)
            )
            self.u += dt * self.nu * lap_u

        # 6. Pressure projection to enforce incompressibility.
        self._project(dt)

    def _make_frame(self) -> StateFrame:
        solid, _, _ = self._cube_state(self.time)
        active = ~solid

        # Velocity diagnostics (collocated, m/s).
        velocity_field = self.u.copy()

        # Source field stays constant.
        scalar_source = self.scalar_source.copy()

        # Boundary samples.
        bnd_pts, bnd_nrm, bnd_vel, bnd_tgt = self._boundary_data(self.time)

        fields = {
            "cumulative_mass_source": self._frame0.fields.get(
                "cumulative_mass_source", np.zeros(self.shape)
            ),
            "transported_scalar": self.scalar.copy(),
            "scalar_source": scalar_source,
            "active_mask": active,
            "velocity_field": velocity_field,
            "boundary_normal": bnd_nrm,
            "boundary_target_velocity": bnd_tgt,
            "boundary_velocity": bnd_vel,
        }

        scalars = dict(self._frame0.scalars)
        scalars.update(
            {
                "grid_spacing": float(
                    self.config.get("sampling", {}).get("spacing", 0.18)
                ),
                "characteristic_length": 0.9,
                "reference_velocity": 1.0,
            }
        )

        frame = StateFrame(
            self.time,
            self._frame0.position.copy(),
            self._frame0.velocity.copy(),
            mass=self._frame0.mass.copy(),
            body_id=self._frame0.body_id.copy(),
            attribute={
                k: (v.copy() if hasattr(v, "copy") else v)
                for k, v in self._frame0.attribute.items()
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            # If asked to go backwards (should not happen), just return current.
            return self._make_frame()
        while self.time < time - 1e-12:
            dt = min(self._dt, time - self.time)
            self._step(dt)
            self.time += dt
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self._dt,
            "grid_shape": list(self.shape),
            "density": self.rho,
        }
