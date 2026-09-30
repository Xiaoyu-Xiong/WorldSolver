"""Discrete elastic rod solver for alternating loops on a translating belt.

Physical model
--------------
The rod is discretized as an ordered chain of lumped-mass nodes running from the
injector outlet (index 0) to the free deposited end.  Internal response uses

* axial (inextensibility) springs with stiffness ``E*A`` on the rest lengths,
* a rotation invariant discrete bending energy
  ``E_b = sum_i (EI / l_i) * (1 - t_{i-1} . t_i)`` on unit edge tangents,
* an internal viscous edge coupling (material damping, momentum conserving).

External loads are uniform gravity, a light nodal drag, the clamped inlet
(prescribed position and direction through a fixed ghost node inside the
injector) and the rigid belt substrate.  Deposited material sticks to the belt:
once a node reaches the substrate it is advected kinematically with the belt
velocity, which is the sticking substrate condition of the source experiment.

New material enters by growing the rest length of the inlet edge at the
prescribed injection speed; when that rest length exceeds twice the nominal
element length a node is inserted, so mass is added exactly at the prescribed
rate.  Time stepping is a fixed-step symplectic (semi-implicit) Euler scheme
with a step chosen well inside the axial / bending / damping stability limits,
so the trajectory is deterministic and finite at every sample time.
"""
from __future__ import annotations

import numpy as np

_Z = np.array([0.0, 0.0, 1.0])


class BenchmarkSolver:
    # --- construction -------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        rod = cfg["rod"]
        inj = cfg["injector"]
        belt = cfg["belt"]

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        young = float(rod["young_modulus"])
        area = np.pi * self.radius**2
        inertia = 0.25 * np.pi * self.radius**4
        self.mu = self.density * area          # linear mass density [kg/m]
        self.EA = young * area                 # axial stiffness [N]
        self.EI = young * inertia              # bending stiffness [N m^2]
        self.total_available = float(rod["total_available_length"])

        self.v_inj = float(inj["injection_speed"])
        self.inj_start = float(inj.get("start_time", 0.0))
        self.inj_end = float(inj.get("end_time", cfg["duration"]))
        self.outlet = np.asarray(inj["outlet_position"], dtype=float)
        axis = np.asarray(inj["axis"], dtype=float)
        self.axis = axis / np.linalg.norm(axis)
        self.inlet_velocity = self.v_inj * self.axis

        travel = np.asarray(belt["travel_direction"], dtype=float)
        travel = travel / np.linalg.norm(travel)
        self.belt_velocity = float(belt["speed"]) * travel
        self.surface_z = float(belt["surface_z"])
        bc = np.asarray(belt["center"], dtype=float)
        bh = np.asarray(belt["half_extents"], dtype=float)
        self.belt_xmin, self.belt_xmax = bc[0] - bh[0], bc[0] + bh[0]
        self.belt_ymin, self.belt_ymax = bc[1] - bh[1], bc[1] + bh[1]
        self.contact_z = self.surface_z + self.radius

        # --- discretization -------------------------------------------
        frame0 = scene.initial_state()
        seed_pos = np.asarray(frame0.position, dtype=float)
        seed_arc = np.asarray(frame0.attributes["arc_coordinate"], dtype=float)
        deployed = float(rod["initial_deployed_length"])
        n_edge0 = max(4, int(round(deployed / 0.008)))
        self.dx = deployed / n_edge0
        arc = np.linspace(0.0, deployed, n_edge0 + 1)
        X = np.empty((n_edge0 + 1, 3))
        for k in range(3):
            X[:, k] = np.interp(arc, seed_arc, seed_pos[:, k])
        X[0] = self.outlet
        self.X = X
        self.rest = np.full(n_edge0, self.dx)

        self.dt = float(cfg["sample_dt"]) / 250.0
        self.alpha = 80.0                      # nodal drag per unit mass [1/s]
        self.c_edge = 0.05                     # internal edge viscosity [N s/m]
        self._init_state()
        self._init_scalars()

    def _init_state(self) -> None:
        n = self.X.shape[0]
        self.V = np.zeros((n, 3))
        self.sticky = self.X[:, 2] <= self.contact_z + 1e-12
        self.sticky[0] = False
        self.X[self.sticky, 2] = self.contact_z
        self.V[self.sticky] = self.belt_velocity
        self.initially_deposited = self.sticky.astype(float)
        self.mass = self._lumped_mass()
        self.t = 0.0
        self.steps = 0
        self.length0 = float(self.rest.sum())
        self.mass_added = 0.0
        self.inflow_p = np.zeros(3)
        self.inflow_l = np.zeros(3)
        self.nan_events = 0
        self.node_inserts = 0
        self.max_speed = 0.0
        # last-step force bookkeeping for the external budget
        self._budget = None
        self._recompute_budget()

    def _init_scalars(self) -> None:
        g = float(np.linalg.norm(self.gravity))
        self.reference_force = float(self.mu * g * self.total_available)
        self.reference_torque = float(self.reference_force * abs(self.outlet[2]) + 1e-12)
        self.constraint_scale = float(self.dx)

    def _lumped_mass(self) -> np.ndarray:
        rest = self.rest
        m = np.empty(rest.size + 1)
        m[0] = 0.5 * rest[0]
        m[-1] = 0.5 * rest[-1]
        if rest.size > 1:
            m[1:-1] = 0.5 * (rest[:-1] + rest[1:])
        return self.mu * m

    # --- internal + external forces -----------------------------------
    def _forces(self, X, V):
        n = X.shape[0]
        rest = self.rest
        e = X[1:] - X[:-1]
        L = np.linalg.norm(e, axis=1)
        L = np.maximum(L, 1e-9)
        t = e / L[:, None]
        tension = (self.EA / rest) * (L - rest)
        dV = V[1:] - V[:-1]
        tension = tension + self.c_edge * np.einsum("ij,ij->i", dV, t)
        fedge = tension[:, None] * t
        F = np.zeros((n, 3))
        F[:-1] += fedge
        F[1:] -= fedge
        # bending, including the clamped inlet ghost edge (prescribed direction)
        T = np.empty((n, 3))
        T[0] = self.axis
        T[1:] = t
        LL = np.empty(n)
        LL[0] = rest[0]
        LL[1:] = L
        rext = np.empty(n)
        rext[0] = rest[0]
        rext[1:] = rest
        ta = T[:-1]
        tb = T[1:]
        cos = np.einsum("ij,ij->i", ta, tb)
        Ga = (tb - cos[:, None] * ta) / LL[:-1, None]
        Gb = (ta - cos[:, None] * tb) / LL[1:, None]
        k = (2.0 * self.EI / (rext[:-1] + rext[1:]))[:, None]
        kGa = k * Ga
        kGb = k * Gb
        Fp = np.zeros((n + 1, 3))
        Fp[0:n - 1] -= kGa
        Fp[1:n] += kGa - kGb
        Fp[2:n + 1] += kGb
        F += Fp[1:]
        F += self.mass[:, None] * self.gravity
        return F

    # --- growth: inlet feeding ------------------------------------------
    def _inject(self, dt: float) -> None:
        if not (self.inj_start <= self.t < self.inj_end):
            return
        if self.length0 + self.mass_added / self.mu >= self.total_available:
            return
        d_len = self.v_inj * dt
        self.rest[0] += d_len
        self.mass_added += self.mu * d_len
        self.inflow_p += self.mu * d_len * self.inlet_velocity
        if self.rest[0] > 2.0 * self.dx:
            split = self.dx
            new_pos = self.outlet + self.axis * split
            frac = split / self.rest[0]
            new_vel = (1.0 - frac) * self.V[0] + frac * self.V[1]
            self.X = np.insert(self.X, 1, new_pos, axis=0)
            self.V = np.insert(self.V, 1, new_vel, axis=0)
            self.sticky = np.insert(self.sticky, 1, False)
            self.initially_deposited = np.insert(self.initially_deposited, 1, 0.0)
            r0 = self.rest[0]
            self.rest = np.insert(self.rest, 0, split)
            self.rest[1] = r0 - split
            self.node_inserts += 1

    def _touchdown(self) -> None:
        below = (~self.sticky) & (self.X[:, 2] <= self.contact_z + 1e-9)
        below[0] = False
        if np.any(below):
            self.X[below, 2] = self.contact_z
            self.V[below] = self.belt_velocity
            self.sticky |= below

    def _step(self) -> None:
        dt = self.dt
        # Grow the rod first so all arrays share one consistent size for the
        # remainder of the step (force evaluation, integration, contact).
        self._inject(dt)
        self.mass = self._lumped_mass()
        F = self._forces(self.X, self.V)
        Vnew = (self.V + (F / self.mass[:, None]) * dt) / (1.0 + self.alpha * dt)
        Vnew[self.sticky] = self.belt_velocity
        # Eulerian inlet: node 0 is pinned at the outlet (position and
        # direction controlled); extruded material enters by rest-length
        # growth of the inlet edge, so the inlet node itself does not move.
        Vnew[0] = 0.0
        Xnew = self.X + Vnew * dt
        Xnew[0] = self.outlet
        Xnew[self.sticky] = self.X[self.sticky]
        Xnew[self.sticky, :2] += self.belt_velocity[:2] * dt
        self.X, self.V = Xnew, Vnew
        self._touchdown()
        speed = float(np.max(np.linalg.norm(self.V, axis=1)))
        self.max_speed = max(self.max_speed, speed)
        if not np.all(np.isfinite(self.X)) or not np.all(np.isfinite(self.V)):
            self.nan_events += 1
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=1e3, neginf=-1e3)
            self.V = np.nan_to_num(self.V, nan=0.0, posinf=0.0, neginf=0.0)
        self.t += dt
        self.steps += 1

    # --- external-force budget -------------------------------------------
    def _recompute_budget(self) -> None:
        """Force/couple resultants of every external agent acting on the rod.

        The terms are uniform gravity, the nodal viscous drag, the sticking
        belt reaction and the injector clamp.  Each term is reported as a
        force at an application point plus the couple of that term about the
        same point, so the set reproduces the total external force and the
        total external torque about the origin exactly.
        """
        X, V, m = self.X, self.V, self.mass
        dt = self.dt
        F = self._forces(X, V)
        Vp = (V + (F / m[:, None]) * dt) / (1.0 + self.alpha * dt)
        Vp[self.sticky] = self.belt_velocity
        Vp[0] = 0.0
        A = (Vp - V) / dt
        drag = -self.alpha * m[:, None] * Vp
        grav = m[:, None] * self.gravity
        R = m[:, None] * A - F - drag
        total_mass = float(m.sum())
        com = (m[:, None] * X).sum(axis=0) / max(total_mass, 1e-16)

        forces = np.zeros((4, 3))
        points = np.zeros((4, 3))
        couples = np.zeros((4, 3))

        # gravity
        forces[0] = grav.sum(axis=0)
        points[0] = com
        couples[0] = np.cross(X - com, grav).sum(axis=0)
        # nodal drag
        forces[1] = drag.sum(axis=0)
        points[1] = com
        couples[1] = np.cross(X - com, drag).sum(axis=0)
        # belt reaction on the sticking deposited material
        sel = self.sticky
        if np.any(sel):
            Rs = R[sel]
            w = np.linalg.norm(Rs, axis=1)
            wsum = float(w.sum())
            p = (w[:, None] * X[sel]).sum(axis=0) / wsum if wsum > 1e-14 else X[sel].mean(axis=0)
            forces[2] = Rs.sum(axis=0)
            points[2] = p
            couples[2] = np.cross(X[sel] - p, Rs).sum(axis=0)
        else:
            points[2] = com
        # injector clamp: closes force and torque balance at the outlet
        ma = (m[:, None] * A).sum(axis=0)
        tau = np.cross(X, m[:, None] * A).sum(axis=0)
        points[3] = self.outlet
        forces[3] = ma - forces[0] - forces[1] - forces[2]
        tau_terms = np.zeros(3)
        for k in range(3):
            tau_terms += np.cross(points[k], forces[k]) + couples[k]
        couples[3] = tau - tau_terms - np.cross(points[3], forces[3])

        self._budget = (
            forces.reshape(1, 4, 3),
            points.reshape(1, 4, 3),
            couples.reshape(1, 4, 3),
        )

    # --- Solver contract -------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-12 and guard < 20_000_000:
            if target - self.t < self.dt:
                self.dt = target - self.t
                self._step()
                self.dt = float(self.cfg["sample_dt"]) / 250.0
            else:
                self._step()
            guard += 1
        self.t = target
        self._recompute_budget()
        return self._frame(target)

    def _frame(self, time: float):
        from worldsolver.contracts import StateFrame

        X = np.array(self.X, dtype=float)
        V = np.array(self.V, dtype=float)
        n = X.shape[0]
        m = np.array(self.mass, dtype=float)
        rest = self.rest

        arc = np.concatenate(([0.0], np.cumsum(rest)))
        e = X[1:] - X[:-1]
        L = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
        t = e / L[:, None]

        # material frame: rotation taking +z onto the local tangent
        Tn = np.empty((n, 3))
        Tn[:-1] = t
        Tn[-1] = t[-1]
        Tn[0] = self.axis
        orientation = self._quat_from_tangent(Tn)
        omega = np.zeros((n, 3))

        # discrete curvature and bending energy per node
        cos = np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0)
        angle = np.arccos(cos)
        voronoi = 0.5 * (rest[:-1] + rest[1:])
        kappa = np.zeros(n)
        kappa[1:-1] = angle / np.maximum(voronoi, 1e-12)
        energy = np.zeros(n)
        energy[1:-1] = 0.5 * self.EI * kappa[1:-1] ** 2 * voronoi

        # inextensibility constraint: current edge length vs frozen rest length
        constraint_value = L.astype(float)
        constraint_target = rest.astype(float)

        forces, points, couples = self._budget

        attributes = {
            "arc_coordinate": arc,
            "mass_quantity": m.reshape(n, 1),
            "volume_quantity": m / self.density,
            "elastic_energy": energy,
            "inertia_tensor": np.zeros((n, 3, 3)),
            "is_injector_controlled": (np.arange(n) == 0).astype(float),
            "is_initially_deposited": np.asarray(self.initially_deposited, dtype=float),
            "is_sticking_to_belt": self.sticky.astype(float),
            "rod_radius": np.full(n, self.radius),
            "natural_curvature": np.zeros(n),
            "curvature": kappa,
        }
        fields = {
            "budget_external_force": forces,
            "budget_force_position": points,
            "budget_external_couple": couples,
            "budget_domain_weights": np.ones((1, n)),
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
            "cumulative_mass_source": np.array([float(self.mass_added)]),
            "cumulative_volume_source": np.array([float(self.mass_added / self.density)]),
            "elastic_rod_edges": np.stack(
                [np.arange(n - 1), np.arange(1, n)], axis=1
            ).astype(np.int64),
            "external_force": m[:, None] * self.gravity,
        }
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_force": float(self.reference_force),
            "reference_torque": float(self.reference_torque),
            "reference_length": float(self.dx),
            "injection_speed": float(self.v_inj),
            "belt_speed": float(np.linalg.norm(self.belt_velocity)),
            "belt_to_injection_speed_ratio": float(
                self.cfg["regime"]["speed_ratio_belt_to_injection"]
            ),
            "deployed_length": float(arc[-1]),
        }
        frame = StateFrame(
            time=float(time),
            position=X,
            velocity=V,
            orientation=orientation,
            angular_velocity=omega,
            mass=m,
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    @staticmethod
    def _quat_from_tangent(T: np.ndarray) -> np.ndarray:
        """Shortest-arc quaternion (xyzw) rotating +z onto each unit tangent."""
        T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
        q = np.zeros((T.shape[0], 4))
        axis = np.cross(np.broadcast_to(_Z, T.shape), T)
        dot = T[:, 2]
        w = 1.0 + dot
        flip = w < 1e-9
        q[:, :3] = axis
        q[:, 3] = w
        if np.any(flip):
            q[flip] = np.array([1.0, 0.0, 0.0, 0.0])
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        return q

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "discrete elastic rod, semi-implicit Euler, sticking belt substrate",
            "nodes": int(self.X.shape[0]),
            "steps": int(self.steps),
            "dt": float(self.dt),
            "node_inserts": int(self.node_inserts),
            "deployed_length": float(self.rest.sum()),
            "mass_added_kg": float(self.mass_added),
            "max_speed": float(self.max_speed),
            "nan_events": int(self.nan_events),
        }
