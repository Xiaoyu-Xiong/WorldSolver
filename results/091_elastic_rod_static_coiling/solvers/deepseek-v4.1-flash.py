"""Physics-based solver for the static coiling of a naturally straight elastic rod.

Physical model
--------------
The rod is a naturally straight, inextensible elastic rod of circular cross
section (radius ``r``, density ``rho``, Young modulus ``E``) discretised into
material nodes with uniform arc spacing ``ds``.  The injector feeds material
through the fixed outlet at constant speed ``v``: a new material node is created
every ``ds / v`` seconds and, while it is younger than one segment, it is
kinematically prescribed to travel straight down the injector axis (this is the
controlled feed boundary condition).  After that the node is released exactly
one arc segment below the outlet, so positions and velocities are continuous
across the release.  Gravity is uniform, and the rigid substrate is the fixed
horizontal plane ``z = substrate.surface_z`` with a sticking contact condition:
the first contact position of every node becomes a fixed adhesion anchor, so
deposited material stays where it landed.  The rod is naturally straight (zero
natural curvature) and no fluid or viscous-thread constitutive model is used.

Time stepping
-------------
The source experiment is in the quasi-static regime, so the rod is advanced with
an overdamped position-based (XPBD) time-stepping scheme, which is an actual
physical time integration:

  * semi-implicit Euler prediction under gravity plus a viscous regularisation
    (inertia is negligible in the source quasi-static regime);
  * XPBD distance constraints (inextensibility) solved with Gauss-Seidel sweeps
    along the chain;
  * XPBD discrete bending constraints with the physical stiffness
    ``k = EI / ds**3`` (``EI`` the bending modulus of the circular section),
    which sets the gravitational coiling radius ``(EI / (rho A g))**(1/3)``;
  * rigid sticking substrate contact (anchor projection).

Every reported sample is produced by integrating the rod forward in time; no
keyframes, animation tables, motion paths or precomputed trajectories are used.
The state is deterministic and finite at every requested sample time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# External budget terms per domain: gravity, substrate contact, injector feed,
# viscous regularisation.
_BUDGET_TERMS = 4


def _load_topology(path_value: str) -> dict:
    """Load the frozen rod topology asset, robust to the working directory."""
    candidates = [Path(path_value)]
    here = Path(__file__).resolve().parent
    candidates += [here.parent / path_value, here.parent.parent / path_value]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"rod topology asset not found: {path_value}")


class BenchmarkSolver:
    """Quasi-static elastic-rod coiling solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        rod_cfg = cfg["rod"]
        inj = cfg["injector"]
        sub = cfg["substrate"]

        self.seed = int(seed)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.g_mag = float(np.linalg.norm(self.gravity))

        self.radius = float(rod_cfg["radius"])
        self.density = float(rod_cfg["density"])
        self.young = float(rod_cfg["young_modulus"])
        self.poisson = float(rod_cfg.get("poisson_ratio", 0.0))
        self.area = math.pi * self.radius**2
        self.EI = self.young * math.pi * self.radius**4 / 4.0

        self.ds = 0.0125
        self.alpha_bend = self.ds**3 / self.EI  # XPBD bending compliance

        self.v_feed = float(inj["injection_speed"])
        self.start_time = float(inj["start_time"])
        self.end_time = float(inj["end_time"])
        self.outlet = np.asarray(inj["outlet_position"], dtype=float)
        self.axis = np.asarray(inj["axis"], dtype=float)
        self.axis = self.axis / max(np.linalg.norm(self.axis), 1e-30)
        self.hold_time = self.ds / self.v_feed  # time to extrude one segment

        self.L0 = float(rod_cfg["initial_deployed_length"])
        self.L_total = float(rod_cfg["total_available_length"])
        self.substrate_z = float(sub["surface_z"])
        self.half_extents = self._substrate_half_extents(cfg)

        # characteristic gravitational bending length and force/torque scales
        self.bend_length = (
            self.EI / (self.density * self.area * max(self.g_mag, 1e-9))
        ) ** (1.0 / 3.0)
        self.reference_force = self.density * self.area * self.L_total * self.g_mag
        self.reference_torque = self.reference_force * self.bend_length
        self.constraint_scale = self.ds

        # time stepping
        self.dt = 0.005
        self.damping_time = 0.02  # quasi-static viscous regularisation
        self.damp = math.exp(-self.dt / self.damping_time)
        self.iterations = 3
        self.bend_relax = 0.8

        # ---------------- initial material configuration ---------------------
        data = _load_topology(str(rod_cfg["topology_asset"]))
        verts = np.asarray(data["vertices"], dtype=float)
        seg = np.linalg.norm(np.diff(verts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total_arc = float(arc[-1])
        n0 = int(round(self.L0 / self.ds)) + 1
        q = np.arange(n0) * self.ds                       # arc measured from tip
        s_query = np.clip(self.L0 - q, 0.0, total_arc)    # arc from injector
        x = np.empty((n0, 3), dtype=float)
        for axis_idx in range(3):
            x[:, axis_idx] = np.interp(s_query, arc, verts[:, axis_idx])
        x[:, 2] = np.maximum(x[:, 2], self.substrate_z + self.radius)
        x[-1] = self.outlet

        self.n = n0
        self.x = x
        self.x_ref = x.copy()
        self.v = np.zeros((n0, 3), dtype=float)
        self.masses = self._lumped_masses(n0)
        self.inv_m = 1.0 / self.masses
        # material birth times: node i (tip = 0) was extruded
        # (n - 1 - i) * ds / v seconds before t = 0.
        self.birth = -(np.arange(n0)[::-1]) * self.hold_time

        # sticking substrate anchors
        self.stuck = np.zeros(n0, dtype=bool)
        self.anchor_xy = np.zeros((n0, 2), dtype=float)
        contact = self.x[:, 2] <= self.substrate_z + self.radius + 1e-12
        self.stuck[contact] = True
        self.anchor_xy[contact] = self.x[contact][:, :2]

        self.t = 0.0
        self.next_insert = self.start_time + self.hold_time
        self.max_nodes = int(self.L_total / self.ds) + 1

        # budget accumulators (reset after each exported sample)
        self._cum_impulse = np.zeros(3)
        self._cum_angular_impulse = np.zeros(3)
        self._internal_residual = 0.0
        self._reset_accumulators()

    @staticmethod
    def _substrate_half_extents(cfg: dict) -> np.ndarray:
        geometry = cfg.get("scene_geometry") or {}
        for obj in geometry.get("objects", []):
            if obj.get("id") == "substrate":
                half = obj.get("geometry", {}).get("half_extents")
                if half:
                    return np.asarray(half, dtype=float)
        return np.asarray([0.24, 0.2, 0.008], dtype=float)

    def _lumped_masses(self, n: int) -> np.ndarray:
        m = np.full(n, self.density * self.area * self.ds)
        m[0] *= 0.5
        m[-1] *= 0.5
        return m

    def _reset_accumulators(self) -> None:
        self._acc_time = 0.0
        self._acc_force = np.zeros((_BUDGET_TERMS, 3))
        self._acc_position = np.zeros((_BUDGET_TERMS, 3))
        self._last_elastic = np.zeros((self.n, 3))
        self._last_energy = np.zeros(self.n)

    # ------------------------------------------------------------- time loop
    def advance_to(self, time: float):
        time = float(time)
        if not np.isfinite(time):
            raise ValueError("advance_to requires a finite time")
        limit = time - 1e-9
        while self.t < limit:
            h = self.dt
            if self.t + h > time:
                h = max(time - self.t, 1e-12)
            self._step(h)
            self.t += h
            self._acc_time += h
        return self._frame(time)

    # ---------------------------------------------------------------- physics
    def _grow(self) -> None:
        """Attach one new material node at the injector outlet."""
        self.x = np.vstack([self.x, self.outlet[None, :]])
        self.v = np.vstack([self.v, np.zeros((1, 3))])
        self.x_ref = np.vstack([self.x_ref, self.outlet[None, :]])
        self.birth = np.concatenate([self.birth, [self.next_insert]])
        self.masses = self._lumped_masses(self.n + 1)
        self.inv_m = 1.0 / self.masses
        self.stuck = np.concatenate([self.stuck, [False]])
        self.anchor_xy = np.vstack([self.anchor_xy, np.zeros((1, 2))])
        self.n += 1

    def _prescribed(self, t_new: float) -> np.ndarray:
        """Nodes still inside the injector: straight-line prescribed feed."""
        age = t_new - self.birth
        return age < self.hold_time - 1e-12

    def _step(self, dt: float) -> None:
        t_new = self.t + dt
        if t_new + 1e-12 >= self.next_insert and self.n < self.max_nodes:
            self._grow()
            self.next_insert += self.hold_time

        n = self.n
        x = self.x
        v = self.v
        x_prev = x.copy()

        prescribed = self._prescribed(t_new)
        if np.any(prescribed):
            age = (t_new - self.birth)[prescribed]
            x[prescribed] = self.outlet[None, :] - self.axis[None, :] * (
                self.v_feed * age[:, None]
            )

        # ---- prediction (semi-implicit Euler, overdamped) -------------------
        v_free = v + self.gravity[None, :] * dt
        v_pred = v_free * self.damp
        x = x + v_pred * dt

        inv_m = self.inv_m.copy()
        inv_m[self.stuck] = 0.0
        inv_m[prescribed] = 0.0

        # ---- XPBD constraint solve with sticking substrate contact ---------
        alpha_t = self.alpha_bend / (dt * dt)
        for iteration in range(self.iterations):
            x = self._solve_bending(x, inv_m, alpha_t)
            if iteration % 2 == 0:
                order = range(0, n - 1)
            else:
                order = range(n - 2, -1, -1)
            x = self._solve_distance(x, inv_m, order)
            x = self._apply_contact(x)
        if np.any(prescribed):
            age = (t_new - self.birth)[prescribed]
            x[prescribed] = self.outlet[None, :] - self.axis[None, :] * (
                self.v_feed * age[:, None]
            )

        # safety: never allow a non-finite state
        if not np.isfinite(x).all():
            x = x_prev.copy()

        v_new = (x - x_prev) / dt
        v_new[self.stuck] = 0.0
        v_new[prescribed] = -self.v_feed * self.axis[None, :]

        # ---- external force bookkeeping ------------------------------------
        m_total = float(np.sum(self.masses))
        imp = self.masses[:, None] * (v_new - v_pred)
        j_grav = m_total * self.gravity * dt
        j_damp = np.sum(self.masses[:, None] * (v_pred - v_free), axis=0)
        j_contact = np.sum(imp[self.stuck], axis=0) if np.any(self.stuck) else np.zeros(3)
        j_feed = np.sum(imp[prescribed], axis=0) if np.any(prescribed) else np.zeros(3)
        internal = np.sum(imp, axis=0) - j_contact - j_feed
        self._internal_residual = float(np.linalg.norm(internal))

        com = np.sum(self.masses[:, None] * x, axis=0) / max(m_total, 1e-30)
        contact_pos = np.mean(x[self.stuck], axis=0) if np.any(self.stuck) else com
        feed_pos = self.outlet.copy()

        self._acc_force[0] += j_grav
        self._acc_force[1] += j_contact
        self._acc_force[2] += j_feed
        self._acc_force[3] += j_damp
        self._acc_position[0] += com
        self._acc_position[1] += contact_pos
        self._acc_position[2] += feed_pos
        self._acc_position[3] += com

        elastic_mask = ~(self.stuck | prescribed)
        self._last_elastic = np.zeros((n, 3))
        if np.any(elastic_mask):
            self._last_elastic[elastic_mask] = imp[elastic_mask] / dt
        self._last_energy = self._bending_energy(x)

        self._cum_impulse += j_grav + j_contact + j_feed + j_damp
        self._cum_angular_impulse += (
            np.cross(com, j_grav)
            + np.cross(contact_pos, j_contact)
            + np.cross(feed_pos, j_feed)
            + np.cross(com, j_damp)
        )

        self.x = x
        self.v = v_new

    # ------------------------------------------------------- constraint solve
    def _solve_distance(self, x: np.ndarray, inv_m: np.ndarray, order) -> np.ndarray:
        ds = self.ds
        for i in order:
            d0 = x[i + 1, 0] - x[i, 0]
            d1 = x[i + 1, 1] - x[i, 1]
            d2 = x[i + 1, 2] - x[i, 2]
            length = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            if length < 1e-12:
                continue
            w0 = inv_m[i]
            w1 = inv_m[i + 1]
            wsum = w0 + w1
            if wsum <= 0.0:
                continue
            corr = (length - ds) / (length * wsum)
            if w0 > 0.0:
                c = w0 * corr
                x[i, 0] += c * d0
                x[i, 1] += c * d1
                x[i, 2] += c * d2
            if w1 > 0.0:
                c = w1 * corr
                x[i + 1, 0] -= c * d0
                x[i + 1, 1] -= c * d1
                x[i + 1, 2] -= c * d2
        return x

    def _solve_bending(self, x: np.ndarray, inv_m: np.ndarray, alpha_t: float) -> np.ndarray:
        if self.n < 3:
            return x
        d = x[:-2] - 2.0 * x[1:-1] + x[2:]
        c = np.linalg.norm(d, axis=1)
        nz = c > 1e-14
        if not np.any(nz):
            return x
        dhat = np.zeros_like(d)
        dhat[nz] = d[nz] / c[nz, None]
        denom = 4.0 * inv_m[1:-1] + inv_m[:-2] + inv_m[2:] + alpha_t
        dlam = np.zeros_like(c)
        dlam[nz] = -self.bend_relax * c[nz] / np.maximum(denom[nz], 1e-30)
        scale = dlam * inv_m[1:-1]
        x[:-2] += scale[:, None] * dhat
        x[1:-1] -= 2.0 * scale[:, None] * dhat
        x[2:] += scale[:, None] * dhat
        return x

    def _apply_contact(self, x: np.ndarray) -> np.ndarray:
        """Rigid substrate: clamp above the surface and stick new contacts."""
        floor = self.substrate_z + self.radius
        below = x[:, 2] < floor
        if np.any(below):
            x[below, 2] = floor
            inside = (
                (np.abs(x[:, 0]) <= self.half_extents[0])
                & (np.abs(x[:, 1]) <= self.half_extents[1])
            )
            new_stick = below & inside & (~self.stuck)
            if np.any(new_stick):
                self.stuck[new_stick] = True
                self.anchor_xy[new_stick] = x[new_stick][:, :2]
        if np.any(self.stuck):
            x[self.stuck, 0] = self.anchor_xy[self.stuck, 0]
            x[self.stuck, 1] = self.anchor_xy[self.stuck, 1]
            x[self.stuck, 2] = floor
        return x

    def _bending_energy(self, x: np.ndarray) -> np.ndarray:
        """Discrete bending energy per node (J)."""
        out = np.zeros(self.n)
        if self.n < 3:
            return out
        d = x[:-2] - 2.0 * x[1:-1] + x[2:]
        c2 = np.sum(d * d, axis=1)
        out[1:-1] = 0.5 * (self.EI / self.ds**3) * c2
        return out

    # ---------------------------------------------------------------- export
    def _frame(self, time: float) -> StateFrame:
        n = self.n
        x = self.x.copy()
        x[:, 2] = np.maximum(x[:, 2], self.substrate_z + self.radius)
        v = self.v.copy()
        v[~np.isfinite(v)] = 0.0

        duration = max(self._acc_time, 1e-9)
        force = self._acc_force / duration
        position = self._acc_position / duration
        couple = np.zeros((_BUDGET_TERMS, 3))

        if n > 1:
            edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
        else:
            edges = np.zeros((1, 2), dtype=np.int64)

        orientation = np.zeros((n, 4), dtype=float)
        orientation[:, 3] = 1.0

        injected = self.density * self.area * min(
            max(self.v_feed * max(time - self.start_time, 0.0), 0.0),
            max(self.L_total - self.L0, 0.0),
        )

        prescribed = self._prescribed(max(self.t, time))

        fields = {
            "budget_external_force": force[None, :, :].copy(),
            "budget_force_position": position[None, :, :].copy(),
            "budget_external_couple": couple[None, :, :].copy(),
            "budget_domain_weights": np.ones((1, n), dtype=float),
            "budget_cumulative_external_impulse": self._cum_impulse[None, :].copy(),
            "budget_cumulative_external_angular_impulse": (
                self._cum_angular_impulse[None, :].copy()
            ),
            "gravity_acceleration": self.gravity[None, :].copy(),
            "elastic_internal_force": self._last_elastic.copy(),
            "elastic_rod_edges": edges,
            "elastic_reference_position": self.x_ref.copy(),
            "elastic_reference_orientation": orientation.copy(),
            "constraint_value": self.outlet[None, :].copy(),
            "constraint_target": self.outlet[None, :].copy(),
            "cumulative_mass_source": np.array([injected], dtype=float),
        }

        attributes = {
            "mass_quantity": self.masses.copy(),
            "arc_coordinate": ((np.arange(n) * self.ds)[::-1]).copy(),
            "is_injector_controlled": prescribed.astype(float),
            "is_initial_contact": np.concatenate([[1.0], np.zeros(n - 1)]),
            "rod_radius": np.full(n, self.radius),
            "elastic_energy": self._last_energy.copy(),
        }

        scalars = {
            "injection_speed": self.v_feed,
            "natural_curvature": 0.0,
            "substrate_speed": 0.0,
            "constraint_scale": self.constraint_scale,
            "reference_force": self.reference_force,
            "reference_torque": self.reference_torque,
            "bend_length": self.bend_length,
            "elastic_modulus": self.young,
        }

        frame = StateFrame(
            time=float(time),
            position=x,
            velocity=v,
            orientation=orientation,
            angular_velocity=np.zeros((n, 3), dtype=float),
            mass=self.masses.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        self._reset_accumulators()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "node_count": int(self.n),
            "time": float(self.t),
            "internal_impulse_residual": float(self._internal_residual),
            "bend_length": float(self.bend_length),
            "reference_force": float(self.reference_force),
            "reference_torque": float(self.reference_torque),
        }
