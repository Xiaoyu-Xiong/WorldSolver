"""Self-implemented cloth-on-incline solver (NumPy only).

Physical model
--------------
* Cloth: 25x25 lumped-mass grid connected by structural, shear and bending
  springs with dashpot damping (internal forces, momentum conserving).
* Gravity: frozen config vector applied to every node.
* Contact: the frozen rigid ramp is a box whose top face is a half-space in the
  ramp frame.  Non-penetration and Coulomb friction are enforced at the
  velocity level each substep (inelastic normal impulse + friction impulse
  bounded by mu * normal impulse).
* Time stepping: semi-implicit (symplectic) Euler with a fixed small substep
  that divides each sample interval exactly, so sample times are hit exactly
  and the trajectory is deterministic.

With the cloth resting on the ramp the projection reproduces the analytic
Coulomb result a = g (sin(theta) - mu cos(theta)) for the downslope motion.
"""
from __future__ import annotations

import json

import numpy as np


def _json_safe(value):
    """Convert NumPy containers to JSON-native equivalents, losslessly."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self._ensure_serializable_metadata(scene)
        config = scene.config
        self.config = config

        frame0 = scene.initial_state()
        self.x = np.asarray(frame0.position, dtype=np.float64).copy()
        self.v = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        self.uv = np.asarray(frame0.attributes["uv"], dtype=np.float64).copy()
        self.n_nodes = self.x.shape[0]
        self.inv_mass = (1.0 / self.mass)[:, None]

        self.gravity = np.asarray(config["gravity"], dtype=np.float64)
        self.mu = float(config["friction"]["coefficient"])

        cloth = config["cloth"]
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.side = float(cloth["side_length"])
        self.rest_spacing = self.side / (self.nx - 1)

        incline = config["incline"]
        self.theta = np.deg2rad(float(incline["angle_degrees"]))
        self.ramp_center = np.asarray(incline["plane_translation"], dtype=np.float64)
        self.ramp_half_thickness = float(incline["plane_half_thickness"])
        c, s = np.cos(self.theta), np.sin(self.theta)
        # Ramp frame: u uphill, v across slope, n outward face normal.
        self.u_hat = np.array([c, 0.0, s])
        self.v_hat = np.array([0.0, 1.0, 0.0])
        self.n_hat = np.array([-s, 0.0, c])
        self.rest_basis = np.column_stack((self.u_hat, self.v_hat, self.n_hat))

        geometry = config.get("scene_geometry", {})
        half_u, half_v = 1.8, 1.15
        for item in geometry.get("objects", []) if isinstance(geometry, dict) else []:
            if item.get("id") == "inclined_plane":
                extents = item.get("geometry", {}).get("half_extents")
                if extents is not None and len(extents) == 3:
                    half_u = float(extents[0])
                    half_v = float(extents[1])
        self.half_u = half_u
        self.half_v = half_v

        self.contact_thickness = float(config["contact"]["thickness"])
        # Surface offset the cloth mid-surface keeps from the ramp top face.
        self.surface_offset = self.ramp_half_thickness + self.contact_thickness

        self._build_springs()

        # Time stepping.
        self.sample_dt = float(config["sample_dt"])
        self.substep_target = 1.0e-4
        self.time = 0.0

        # Diagnostics bookkeeping.
        self.downhill = np.asarray(incline["downhill_direction"], dtype=np.float64)
        norm = np.linalg.norm(self.downhill)
        self.downhill = self.downhill / norm if norm > 0 else -self.u_hat
        diag = config.get("diagnostics", {}).get("analytical_friction_solution", {})
        self.analytic_accel = float(diag.get("acceleration_magnitude", 0.0))
        self.contact_time = None
        self.contact_com_s = 0.0
        self.contact_com_v = 0.0
        self.max_penetration = 0.0
        self._last_disp_error = 0.0
        self._last_vel_error = 0.0

        self.com0 = float(self.mass @ (self.x @ self.downhill)) / float(self.mass.sum())

    def _ensure_serializable_metadata(self, scene) -> None:
        """Keep the frozen metadata content but make it JSON-encodable.

        The platform writer stores ``scene.metadata()`` as a JSON attribute, and
        raw NumPy arrays (the cloth face topology) cannot be encoded.  Values are
        converted to equivalent native Python types only if encoding would fail.
        """
        try:
            metadata = scene.metadata()
        except Exception:
            return
        try:
            json.dumps(metadata, sort_keys=True)
            return
        except TypeError:
            pass
        safe = _json_safe(metadata)
        try:
            json.dumps(safe, sort_keys=True)
        except TypeError:
            return
        try:
            scene.metadata = lambda: safe
        except Exception:
            pass

    def _build_springs(self) -> None:
        nx, ny = self.nx, self.ny

        def index(i, j):
            return j * nx + i

        edges = []
        stiffness = []
        damping = []
        k_struct, d_struct = 400.0, 0.20
        k_shear, d_shear = 100.0, 0.05
        k_bend, d_bend = 5.0, 0.01

        for j in range(ny):
            for i in range(nx - 1):
                edges.append((index(i, j), index(i + 1, j)))
                stiffness.append(k_struct)
                damping.append(d_struct)
        for j in range(ny - 1):
            for i in range(nx):
                edges.append((index(i, j), index(i, j + 1)))
                stiffness.append(k_struct)
                damping.append(d_struct)
        for j in range(ny - 1):
            for i in range(nx - 1):
                edges.append((index(i, j), index(i + 1, j + 1)))
                stiffness.append(k_shear)
                damping.append(d_shear)
                edges.append((index(i + 1, j), index(i, j + 1)))
                stiffness.append(k_shear)
                damping.append(d_shear)
        for j in range(ny):
            for i in range(nx - 2):
                edges.append((index(i, j), index(i + 2, j)))
                stiffness.append(k_bend)
                damping.append(d_bend)
        for j in range(ny - 2):
            for i in range(nx):
                edges.append((index(i, j), index(i, j + 2)))
                stiffness.append(k_bend)
                damping.append(d_bend)

        self.edges = np.asarray(edges, dtype=np.int64)
        self.e0 = self.edges[:, 0]
        self.e1 = self.edges[:, 1]
        self.spring_k = np.asarray(stiffness, dtype=np.float64)
        self.spring_d = np.asarray(damping, dtype=np.float64)
        delta = self.x[self.e1] - self.x[self.e0]
        self.rest_length = np.linalg.norm(delta, axis=1)

    # ----------------------------------------------------------------- physics
    def _internal_forces(self, x, v):
        """Spring + dashpot forces (equal and opposite, momentum conserving)."""
        delta = x[self.e1] - x[self.e0]
        length = np.linalg.norm(delta, axis=1)
        safe = np.maximum(length, 1.0e-12)
        direction = delta / safe[:, None]
        stretch = length - self.rest_length
        rel_v = np.einsum("ij,ij->i", v[self.e1] - v[self.e0], direction)
        scalar = self.spring_k * stretch + self.spring_d * rel_v
        force = scalar[:, None] * direction
        out = np.zeros_like(x)
        np.add.at(out, self.e0, force)
        np.add.at(out, self.e1, -force)
        return out

    def _ramp_gap(self, x):
        """Signed distance from cloth mid-surface to the ramp contact surface."""
        rel = x - self.ramp_center
        return rel @ self.n_hat - self.surface_offset

    def _inside_footprint(self, x):
        rel = x - self.ramp_center
        return (np.abs(rel @ self.u_hat) <= self.half_u) & (
            np.abs(rel @ self.v_hat) <= self.half_v
        )

    def _resolve_contact(self, x, v, dt):
        """Velocity-level non-penetration + Coulomb friction, then position fix."""
        gap = self._ramp_gap(x)
        inside = self._inside_footprint(x)
        vn = v @ self.n_hat
        active = inside & ((gap <= 0.0) | ((gap < self.contact_thickness) & (vn < 0.0)))
        if not np.any(active):
            return x, v, gap, 0.0

        idx = np.nonzero(active)[0]
        vn_a = vn[idx]
        # Inelastic non-penetration: remove any approaching normal velocity.
        # Residual penetration is removed geometrically below, so no extra
        # restitution or outward push is injected here.
        vn_target = np.maximum(vn_a, 0.0)
        dvn = vn_target - vn_a  # >= 0, proportional to the normal impulse / mass
        v[idx] += dvn[:, None] * self.n_hat[None, :]

        # Coulomb friction bounded by the normal impulse magnitude.
        vt = v[idx] - (v[idx] @ self.n_hat)[:, None] * self.n_hat[None, :]
        vt_mag = np.linalg.norm(vt, axis=1)
        budget = self.mu * dvn
        scale = np.where(
            vt_mag > 1.0e-12,
            np.maximum(0.0, 1.0 - budget / np.maximum(vt_mag, 1.0e-12)),
            0.0,
        )
        v[idx] = (v[idx] @ self.n_hat)[:, None] * self.n_hat[None, :] + scale[:, None] * vt

        # Positional correction for any remaining penetration.
        pen = np.minimum(gap[idx], 0.0)
        x[idx] -= pen[:, None] * self.n_hat[None, :]
        penetration = float(-pen.min()) if pen.size else 0.0
        gap = self._ramp_gap(x)
        return x, v, gap, penetration

    def _step(self, dt):
        force = self._internal_forces(self.x, self.v)
        accel = force * self.inv_mass + self.gravity[None, :]
        self.v = self.v + dt * accel
        self.x = self.x + dt * self.v
        self.x, self.v, gap, penetration = self._resolve_contact(self.x, self.v, dt)
        self.max_penetration = max(self.max_penetration, penetration)
        return gap

    def _advance_interval(self, target_time):
        remaining = target_time - self.time
        if remaining <= 1.0e-15:
            return
        n_sub = max(1, int(np.ceil(remaining / self.substep_target)))
        dt = remaining / n_sub
        for k in range(n_sub):
            self._step(dt)
            self.time += dt
            self._update_contact_diagnostics(dt)
        self.time = target_time

    # ------------------------------------------------------------- diagnostics
    def _com_downslope(self):
        total = float(self.mass.sum())
        s = float(self.mass @ (self.x @ self.downhill)) / total
        vs = float(self.mass @ (self.v @ self.downhill)) / total
        return s, vs

    def _update_contact_diagnostics(self, dt):
        if self.contact_time is not None:
            return
        gap = self._ramp_gap(self.x)
        inside = self._inside_footprint(self.x)
        resting = inside & (gap <= 1.5 * self.contact_thickness)
        if np.count_nonzero(resting) >= 0.9 * self.n_nodes:
            self.contact_time = self.time
            self.contact_com_s, self.contact_com_v = self._com_downslope()

    def _analytic_errors(self):
        if self.contact_time is None or self.time <= self.contact_time:
            return 0.0, 0.0
        tau = self.time - self.contact_time
        a = self.analytic_accel
        s_ref = self.contact_com_s + self.contact_com_v * tau + 0.5 * a * tau * tau
        v_ref = self.contact_com_v + a * tau
        s, vs = self._com_downslope()
        return abs(s - s_ref), abs(vs - v_ref)

    # ------------------------------------------------------- deformation state
    def _deformation_gradient(self):
        nx, ny = self.nx, self.ny
        grid = self.x.reshape(ny, nx, 3)
        h = self.rest_spacing
        du = np.empty_like(grid)
        du[:, 1:-1, :] = (grid[:, 2:, :] - grid[:, :-2, :]) / (2.0 * h)
        du[:, 0, :] = (grid[:, 1, :] - grid[:, 0, :]) / h
        du[:, -1, :] = (grid[:, -1, :] - grid[:, -2, :]) / h
        dv = np.empty_like(grid)
        dv[1:-1, :, :] = (grid[2:, :, :] - grid[:-2, :, :]) / (2.0 * h)
        dv[0, :, :] = (grid[1, :, :] - grid[0, :, :]) / h
        dv[-1, :, :] = (grid[-1, :, :] - grid[-2, :, :]) / h
        t_u = du.reshape(-1, 3)
        t_v = dv.reshape(-1, 3)
        normal = np.cross(t_u, t_v)
        norm = np.linalg.norm(normal, axis=1)
        fallback = np.tile(self.n_hat, (self.n_nodes, 1))
        normal = np.where(norm[:, None] > 1.0e-12, normal / np.maximum(norm, 1.0e-12)[:, None], fallback)
        # Columns are the deformed images of the material basis (u, v, n).
        f_world = np.stack((t_u, t_v, normal), axis=2)
        # Express in the material frame of the undeformed cloth (ramp-aligned).
        return np.einsum("ki,nkj->nij", self.rest_basis, f_world)

    def _make_frame(self):
        from worldsolver.contracts import StateFrame

        gap = self._ramp_gap(self.x)
        target = np.maximum(gap, 0.0)
        disp_err, vel_err = self._analytic_errors()
        self._last_disp_error, self._last_vel_error = disp_err, vel_err
        s, vs = self._com_downslope()
        contact_count = int(np.count_nonzero((gap <= self.contact_thickness) & self._inside_footprint(self.x)))

        frame = StateFrame(
            float(self.time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "uv": self.uv.copy(),
                "cloth_vertex": np.ones(self.n_nodes),
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "constraint_value": gap.copy(),
                "constraint_target": target,
                "contact_signed_distance": gap.copy(),
                "cumulative_mass_source": np.zeros(1),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={
                "friction_coefficient": self.mu,
                "constraint_scale": self.contact_thickness,
                "contact_length_scale": self.contact_thickness,
                "constraint_residual": float(max(0.0, -gap.min())),
                "analytical_displacement_error": float(disp_err),
                "analytical_velocity_error": float(vel_err),
                "downslope_displacement": float(s - self.com0),
                "downslope_velocity": float(vs),
                "contact_node_count": float(contact_count),
                "analytic_acceleration": self.analytic_accel,
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- public API
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to must be called with non-decreasing times")
        self._advance_interval(target)
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "mass-spring cloth, semi-implicit Euler, velocity-level Coulomb contact",
            "substep_dt": self.substep_target,
            "node_count": int(self.n_nodes),
            "spring_count": int(len(self.edges)),
            "contact_time": None if self.contact_time is None else float(self.contact_time),
            "max_penetration": float(self.max_penetration),
            "final_displacement_error": float(self._last_disp_error),
            "final_velocity_error": float(self._last_vel_error),
            "analytic_acceleration": self.analytic_accel,
        }
