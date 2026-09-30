"""Solver Agent-owned implementation stub.

Implement ``BenchmarkSolver`` in this file.

The runner calls::

    solver.initialize(scene, seed)
    frames = [solver.advance_to(float(t)) for t in scene.sample_times()]

``advance_to(time)`` must return ``worldsolver.contracts.StateFrame`` with
finite state arrays and all trajectory fields requested in the Solver prompt.
The class must define ``initialize``, ``advance_to``, and ``diagnostics``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Mass-spring cloth plus point/rigid dinnerware tablecloth-pull solver.

    The cloth is modelled as a triangular mass-spring network.  Rigid bodies
    are approximated by their centres of mass and simple contact proxies; the
    cloth drags them through a velocity-spring coupling.  Integration is a
    stable explicit Euler substepping scheme with table contact and a pulled
    edge constraint.
    """

    def initialize(self, scene, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.config = getattr(scene, "config", {}) or {}

        # ---- scalar parameters ------------------------------------------------
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.pull_speed = float(self.config.get("pull_speed", 2.0))
        self.pull_duration = float(self.config.get("pull_duration", 0.9))
        self.table_top = float(self.config.get("table_top", 0.78))
        self.max_dt = 1e-4

        cloth_cfg = self.config.get("cloth", {})
        self.cloth_thickness = float(cloth_cfg.get("thickness", 0.008))
        self.cloth_total_mass = float(cloth_cfg.get("total_mass", 0.48))
        self.rigid_cfg = {int(rb["body_id"]): rb for rb in self.config.get("rigid_bodies", [])}

        # ---- load frozen initial state ---------------------------------------
        root = Path(__file__).resolve().parent.parent / "assets"
        init = json.loads((root / "initial_state.json").read_text(encoding="utf-8"))
        meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

        self.position = np.asarray(init["position"], dtype=float).copy()
        self.velocity = np.asarray(init["velocity"], dtype=float).copy()
        self.N = self.position.shape[0]

        # body_id (fall back to a sensible partition if missing)
        if "body_id" in init:
            self.body_id = np.asarray(init["body_id"], dtype=int)
        else:
            n_rigid = len(self.rigid_cfg)
            n_cloth = self.N - n_rigid
            self.body_id = np.concatenate(
                [np.zeros(max(0, n_cloth), dtype=int), np.arange(1, n_rigid + 1, dtype=int)]
            )[: self.N]
        self.body_id = np.asarray(self.body_id, dtype=int)

        # cloth / rigid partition used by mass init and topology below
        self.cloth_mask = self.body_id == 0
        self.cloth_idx = np.where(self.cloth_mask)[0]

        # mass per sample
        self.mass = self._init_mass(init.get("mass"))

        # orientation / angular velocity (kept frozen; rotation not simulated)
        if "orientation" in init and np.asarray(init["orientation"]).shape == (self.N, 4):
            self.orientation = np.asarray(init["orientation"], dtype=float).copy()
        else:
            self.orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (self.N, 1))
        if "angular_velocity" in init and np.asarray(init["angular_velocity"]).shape == (self.N, 3):
            self.angular_velocity = np.asarray(init["angular_velocity"], dtype=float).copy()
        else:
            self.angular_velocity = np.zeros((self.N, 3), dtype=float)

        # ---- cloth topology ----------------------------------------------------
        faces = np.asarray(meta["topology"]["faces"], dtype=int)
        self.edges = self._build_edge_list(faces)
        self.rest_length = np.linalg.norm(
            self.position[self.edges[:, 1]] - self.position[self.edges[:, 0]], axis=1
        )

        self.cloth_mask = self.body_id == 0
        self.cloth_idx = np.where(self.cloth_mask)[0]
        cloth_edge_mask = self.cloth_mask[self.edges[:, 0]] & self.cloth_mask[self.edges[:, 1]]
        self.cloth_edges = self.edges[cloth_edge_mask]
        self.cloth_rest = self.rest_length[cloth_edge_mask]

        # ---- geometry helpers --------------------------------------------------
        # contact half-extent in z for each sample
        self.contact_z = np.zeros(self.N, dtype=float)
        self.contact_z[self.cloth_mask] = self.cloth_thickness * 0.5
        self._rigid_geom: dict[int, dict] = {}
        self._init_rigid_geometry()

        # ---- pull constraint ---------------------------------------------------
        # Pull the cloth edge at the minimum-y side in the -y direction.
        self.pull_dir = np.array([0.0, -1.0, 0.0], dtype=float)
        self.pull_vel = self.pull_speed * self.pull_dir
        cloth_y = self.position[self.cloth_idx, 1]
        y_min = float(cloth_y.min()) if cloth_y.size else 0.0
        self.pull_mask = np.zeros(self.N, dtype=bool)
        if self.cloth_idx.size:
            near = np.abs(self.position[self.cloth_idx, 1] - y_min) < 1e-4
            self.pull_mask[self.cloth_idx[near]] = True

        # ---- material / numerical parameters -----------------------------------
        # cloth spring stiffness chosen to keep explicit substepping stable
        # with max_dt = 1e-4 while producing visible resistance to stretching.
        self.k_cloth = 8.0e3  # N/m
        self.c_cloth = 0.5    # viscous damping along springs
        self.k_table = 5.0e4  # table normal penalty
        self.c_table = 2.0    # tangential table friction coefficient-ish
        self.mu_table = 0.4   # friction multiplier
        self.k_drag = 20.0    # cloth -> rigid velocity coupling
        self.vdamp = 0.5      # global velocity damping for stability

        self.time = 0.0

    # -------------------------------------------------------------------------
    def _init_mass(self, mass_data) -> np.ndarray:
        if mass_data is not None:
            m = np.asarray(mass_data, dtype=float)
            if m.shape == (self.N,):
                return m
        # compute from configuration
        mass = np.zeros(self.N, dtype=float)
        n_cloth = int(self.cloth_mask.sum())
        per_cloth = self.cloth_total_mass / max(1, n_cloth)
        mass[self.cloth_mask] = per_cloth
        for bid, rb in self.rigid_cfg.items():
            mask = self.body_id == int(bid)
            mass[mask] = float(rb.get("mass", 1.0))
        mass[mass <= 0.0] = 1e-6
        return mass

    def _build_edge_list(self, faces: np.ndarray) -> np.ndarray:
        if faces.size == 0:
            return np.zeros((0, 2), dtype=int)
        e0 = faces[:, [0, 1]]
        e1 = faces[:, [1, 2]]
        e2 = faces[:, [0, 2]]
        edges = np.concatenate([e0, e1, e2], axis=0)
        # canonical order
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        return edges

    def _init_rigid_geometry(self) -> None:
        # Mapping from body_id to simple collision parameters.
        # These mirror the frozen scene_geometry.yaml.
        geom = {
            1: {"type": "cylinder", "radius": 0.085, "height": 0.72},
            2: {"type": "cylinder", "radius": 0.07, "height": 0.30},
            3: {"type": "cylinder", "radius": 0.24, "height": 0.035},
            4: {"type": "box", "half_extents": [0.19, 0.11, 0.035]},
            5: {"type": "box", "half_extents": [0.16, 0.025, 0.012]},
            6: {"type": "box", "half_extents": [0.19, 0.022, 0.012]},
        }
        for bid, info in geom.items():
            mask = self.body_id == int(bid)
            if not mask.any():
                continue
            kind = info["type"]
            if kind == "cylinder":
                z_half = float(info["height"]) * 0.5
                r = float(info["radius"])
            else:  # box
                he = np.array(info["half_extents"], dtype=float)
                z_half = he[2]
                r = np.linalg.norm(he[:2]) * 0.5 + 0.02
            self.contact_z[mask] = z_half
            self._rigid_geom[int(bid)] = {
                "z_half": z_half,
                "radius": float(r),
                "mass": float(self.rigid_cfg.get(int(bid), {}).get("mass", 1.0)),
            }

    # -------------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            return self._make_frame(self.time)

        dt = float(time - self.time)
        steps = max(1, int(np.ceil(dt / self.max_dt)))
        h = dt / steps

        t = self.time
        for _ in range(steps):
            self._substep(h, t)
            t += h

        self.time = float(time)
        return self._make_frame(self.time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit_mass_spring_cloth",
            "samples": self.N,
            "cloth_vertices": int(self.cloth_mask.sum()),
            "substep_dt": self.max_dt,
        }

    # -------------------------------------------------------------------------
    def _substep(self, h: float, t: float) -> None:
        force = np.zeros((self.N, 3), dtype=float)

        # gravity
        force += self.mass[:, None] * self.gravity

        # cloth internal spring forces
        self._add_cloth_forces(force)

        # table contact / friction
        self._add_table_forces(force)

        # cloth -> rigid drag coupling
        self._add_drag_forces(force)

        # explicit velocity update
        inv_m = 1.0 / self.mass
        self.velocity += h * force * inv_m[:, None]

        # mild global damping to suppress ringing
        self.velocity *= (1.0 - self.vdamp * h)

        # pulled edge kinematics
        pull_active = t < self.pull_duration
        if pull_active:
            self.velocity[self.pull_mask] = self.pull_vel

        # position update
        self.position += h * self.velocity

        # hard floor / penetration recovery
        self._enforce_floor()

    def _add_cloth_forces(self, force: np.ndarray) -> None:
        if self.cloth_edges.size == 0:
            return
        i = self.cloth_edges[:, 0]
        j = self.cloth_edges[:, 1]
        dx = self.position[j] - self.position[i]
        dv = self.velocity[j] - self.velocity[i]
        lengths = np.linalg.norm(dx, axis=1)
        # guard division
        safe = lengths > 1e-9
        dirs = np.zeros_like(dx)
        dirs[safe] = dx[safe] / lengths[safe, None]

        spring = self.k_cloth * (lengths - self.cloth_rest)
        dashpot = self.c_cloth * np.einsum("ij,ij->i", dv, dirs)
        mag = spring + dashpot
        fvec = mag[:, None] * dirs

        np.add.at(force, i, fvec)
        np.add.at(force, j, -fvec)

    def _add_table_forces(self, force: np.ndarray) -> None:
        signed = self.position[:, 2] - (self.table_top + self.contact_z)
        penetrating = signed < 0.0
        if not penetrating.any():
            return
        idx = np.where(penetrating)[0]
        # normal penalty
        fn = -self.k_table * signed[idx]
        force[idx, 2] += fn
        # tangential Coulomb-ish friction
        vtan = self.velocity[idx, :2]
        vnorm = np.linalg.norm(vtan, axis=1, keepdims=True)
        vnorm_safe = np.where(vnorm < 1e-6, 1e-6, vnorm)
        # scale friction with normal force magnitude
        friction_mag = self.mu_table * fn[:, None]
        force[idx, :2] -= friction_mag * (vtan / vnorm_safe)

    def _add_drag_forces(self, force: np.ndarray) -> None:
        if self.cloth_idx.size == 0:
            return
        cloth_pos = self.position[self.cloth_idx]
        cloth_vel = self.velocity[self.cloth_idx]
        for idx in np.where(~self.cloth_mask)[0]:
            bid = int(self.body_id[idx])
            geom = self._rigid_geom.get(bid)
            if geom is None:
                continue
            center = self.position[idx]
            radius = geom["radius"]
            # vertices whose horizontal projection lies under the object
            dxy = cloth_pos[:, :2] - center[:2]
            mask = np.linalg.norm(dxy, axis=1) < radius
            if not mask.any():
                continue
            v_avg = cloth_vel[mask].mean(axis=0)
            f = self.k_drag * (v_avg - self.velocity[idx])
            force[idx] += f
            # equal and opposite on the cloth patch
            reaction = -f / mask.sum()
            force[self.cloth_idx[mask]] += reaction

    def _enforce_floor(self) -> None:
        limit = self.table_top + self.contact_z
        below = self.position[:, 2] < limit
        if not below.any():
            return
        self.position[below, 2] = limit[below] + 1e-6
        # kill downward velocity on the floor
        downward = below & (self.velocity[:, 2] < 0.0)
        self.velocity[downward, 2] = 0.0

    # -------------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        # required fields
        contact_dist = self.position[:, 2] - (self.table_top + self.contact_z)
        ext_force = self._external_force()
        cur_len, tgt_len = self._constraint_data()

        attributes = {
            "mass_quantity": self.mass.copy(),
        }
        fields = {
            "contact_signed_distance": contact_dist.astype(float),
            "external_force": ext_force.astype(float),
            "constraint_value": cur_len.astype(float),
            "constraint_target": tgt_len.astype(float),
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        scalars = {
            "constraint_scale": 1.0,
            "contact_length_scale": float(self.cloth_thickness),
        }

        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def _external_force(self) -> np.ndarray:
        """Return external (non internal-spring) forces for the current state."""
        force = np.zeros((self.N, 3), dtype=float)
        force += self.mass[:, None] * self.gravity
        self._add_table_forces(force)
        self._add_drag_forces(force)
        # include the kinematic pull as an constraint force on the pulled edge
        if self.time < self.pull_duration:
            force[self.pull_mask] += 100.0 * (self.pull_vel - self.velocity[self.pull_mask])
        return force

    def _constraint_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Current spring length and rest length for cloth edges."""
        if self.cloth_edges.size == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        dx = self.position[self.cloth_edges[:, 1]] - self.position[self.cloth_edges[:, 0]]
        cur = np.linalg.norm(dx, axis=1)
        return cur, self.cloth_rest.copy()
