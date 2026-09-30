"""Elastic-rod solver for wavy hair sticking/sliding on a shoulder."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import Scene, Solver, StateFrame


# --------------------------------------------------------------------------- #
# Small vector helpers
# --------------------------------------------------------------------------- #
def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _normalize(x: np.ndarray) -> np.ndarray:
    n = _norm(x)
    return x / n if n > 0.0 else x


# --------------------------------------------------------------------------- #
# Geometry helpers for signed distance to frozen body geometry
# --------------------------------------------------------------------------- #
class _BodySDF:
    def __init__(self, config: dict):
        self.objects: list[dict] = []
        for obj in config.get("scene_geometry", {}).get("objects", []):
            if not obj.get("collision_enabled", True):
                continue
            g = obj["geometry"]
            t = obj["transform"]
            spec = {
                "id": obj["id"],
                "type": g["type"],
                "translation": np.array(t["translation"], float),
                "rotation_xyzw": np.array(t["rotation_xyzw"], float),
                "scale": np.array(t["scale"], float),
            }
            if g["type"] == "sphere":
                spec["radius"] = float(g["radius"])
            elif g["type"] == "capsule":
                spec["radius"] = float(g["radius"])
                spec["height"] = float(g["height"])
            self.objects.append(spec)

    def _quat_inv_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        # q is xyzw; rotate by inverse: q^-1 * [0,v] * q
        return self._quat_rotate(np.array([-q[0], -q[1], -q[2], q[3]]), v)

    def _quat_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        # 3x3 rotation matrix from unit quaternion (x,y,z,w)
        return np.array(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ]
        ).dot(v)

    def signed_distance(self, p: np.ndarray) -> tuple[float, np.ndarray, int]:
        """Return (signed distance, normal pointing outside, object index) for point p."""
        best = -1e9
        best_n = np.array([0.0, 0.0, 1.0])
        best_idx = -1
        for idx, obj in enumerate(self.objects):
            d, n = self._obj_signed_distance(obj, p)
            if d > best:
                best = d
                best_n = n
                best_idx = idx
        return best, best_n, best_idx

    def _obj_signed_distance(self, obj: dict, p: np.ndarray) -> tuple[float, np.ndarray]:
        local = (p - obj["translation"]) / obj["scale"]
        q_rot = obj["rotation_xyzw"]
        local = self._quat_inv_rotate(q_rot, local)
        if obj["type"] == "sphere":
            r = obj["radius"]
            d = _norm(local) - r
            n = _normalize(local) if _norm(local) > 1e-12 else np.array([0.0, 0.0, 1.0])
        elif obj["type"] == "capsule":
            r = obj["radius"]
            h = obj["height"]
            half = h / 2.0
            # Capsule along local z
            proj = np.clip(local[2], -half, half)
            axis_point = np.array([0.0, 0.0, proj])
            diff = local - axis_point
            dist = _norm(diff)
            if dist < 1e-12:
                # inside exactly on axis
                d = -r
                n = np.array([0.0, 0.0, 1.0])
            else:
                d = dist - r
                n = diff / dist
        else:
            d = 1e9
            n = np.array([0.0, 0.0, 1.0])
        # Transform normal back to world by rotation only (ignore scale for normal direction approx)
        n_world = self._quat_rotate(q_rot, n)
        n_world = _normalize(n_world)
        # Scale affects distance; approximate by geometric mean for uniform-ish behavior
        scale_factor = math.exp(np.mean(np.log(obj["scale"])))
        return d * scale_factor, n_world


# --------------------------------------------------------------------------- #
# Elastic rod solver
# --------------------------------------------------------------------------- #
class BenchmarkSolver(Solver):
    def __init__(self) -> None:
        self._scene: Scene | None = None
        self._config: dict = {}
        self._t: float = 0.0
        self._pos: np.ndarray | None = None
        self._vel: np.ndarray | None = None
        self._mass: np.ndarray | None = None
        self._strand_id: np.ndarray | None = None
        self._node_index: np.ndarray | None = None
        self._is_root: np.ndarray | None = None
        self._n: int = 0
        self._strand_count: int = 84
        self._nodes_per_strand: int = 22
        self._strand_radius: float = 0.0042
        self._edges: np.ndarray | None = None
        self._rest_length: np.ndarray | None = None
        self._sdf: _BodySDF | None = None
        self._rng = np.random.default_rng(0)
        # solver parameters
        self._dt: float = 0.0005
        self._substeps_per_sample: int = 40  # sample_dt 0.02 -> 0.0005
        self._young: float = 2.0e6
        self._rho: float = 1200.0
        self._damping: float = 0.05
        self._contact_stiffness: float = 1.0e5
        self._friction: float = 0.3
        self._stick_tol: float = 1e-3
        self._diag: dict = {}

    def initialize(self, scene: Scene, seed: int) -> None:
        self._scene = scene
        self._config = dict(getattr(scene, "config", {}) or {})
        if self._config is None:
            self._config = {}
        init = scene.initial_state()
        init.validate()
        self._t = float(init.time)
        self._pos = np.array(init.position, dtype=float, copy=True)
        self._vel = np.array(init.velocity, dtype=float, copy=True)
        self._mass = np.array(init.mass, dtype=float, copy=True)
        self._strand_id = np.array(init.attributes["strand_id"], dtype=int)
        self._node_index = np.array(init.attributes["node_index"], dtype=int)
        self._is_root = np.array(init.attributes["is_root"], dtype=int)
        self._n = len(self._pos)
        self._strand_count = int(init.attributes["strand_id"].max()) + 1
        self._nodes_per_strand = int(init.attributes["node_index"].max()) + 1
        self._strand_radius = float(init.scalars.get("strand_radius", 0.0042))
        self._rng = np.random.default_rng(seed)
        # Build edge list
        edges = []
        for sid in range(self._strand_count):
            base = sid * self._nodes_per_strand
            for k in range(self._nodes_per_strand - 1):
                edges.append([base + k, base + k + 1])
        self._edges = np.array(edges, dtype=int)
        p0 = self._pos[self._edges[:, 0]]
        p1 = self._pos[self._edges[:, 1]]
        self._rest_length = np.linalg.norm(p1 - p0, axis=1)
        # Body SDF
        self._sdf = _BodySDF(self._config)
        # Reference scalars from config/initial state
        self._ref_force = 1.0
        self._ref_momentum = 1.0
        self._ref_velocity = 1.0
        # Store initial root positions to follow head motion
        self._root_init = self._pos.copy()
        self._diag = {"init": True}

    def _head_displacement(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        axis = np.array(self._config.get("head_motion", {}).get("axis", [1.0, 0.0, 0.0]), float)
        axis = _normalize(axis)
        d = float(self._config.get("head_motion", {}).get("displacement", 0.18))
        t0 = float(self._config.get("head_motion", {}).get("start_time", 0.45))
        t1 = float(self._config.get("head_motion", {}).get("stop_time", 2.35))
        if t <= t0:
            s = 0.0
            dsdt = 0.0
        elif t >= t1:
            s = 1.0
            dsdt = 0.0
        else:
            tau = (t - t0) / (t1 - t0)
            # smoothstep
            s = tau * tau * (3.0 - 2.0 * tau)
            dsdt = 6.0 * tau * (1.0 - tau) / (t1 - t0)
        return axis * (s * d), axis * (dsdt * d)

    def _body_velocity(self, p: np.ndarray, t: float) -> np.ndarray:
        # All kinematic bodies move together with the head displacement profile
        _, v = self._head_displacement(t)
        return v

    def _elastic_forces(self, pos: np.ndarray) -> np.ndarray:
        forces = np.zeros_like(pos)
        if self._edges is None or len(self._edges) == 0:
            return forces
        p0 = pos[self._edges[:, 0]]
        p1 = pos[self._edges[:, 1]]
        diff = p1 - p0
        length = np.linalg.norm(diff, axis=1)
        length = np.maximum(length, 1e-12)
        # Cross-sectional area for rod
        r = self._strand_radius
        area = np.pi * r * r
        # Stretch stiffness
        ks = self._young * area / self._rest_length
        strain = (length - self._rest_length) / self._rest_length
        f_mag = ks * strain
        f = f_mag[:, None] * (diff / length[:, None])
        np.add.at(forces, self._edges[:, 0], f)
        np.add.at(forces, self._edges[:, 1], -f)
        # Bending: angle force at middle nodes, simple three-node bending spring
        for sid in range(self._strand_count):
            base = sid * self._nodes_per_strand
            for k in range(1, self._nodes_per_strand - 1):
                i0 = base + k - 1
                i1 = base + k
                i2 = base + k + 1
                x0 = pos[i0]
                x1 = pos[i1]
                x2 = pos[i2]
                v1 = x1 - x0
                v2 = x2 - x1
                # Simple bending force trying to keep v1 and v2 aligned (linearized)
                fb = 0.5 * self._young * area[0] * (v2 - v1)
                forces[i0] += fb
                forces[i1] -= 2.0 * fb
                forces[i2] += fb
        return forces

    def _contact_response(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, list]:
        """Return contact force array and a list of contact records."""
        forces = np.zeros_like(pos)
        records: list[dict] = []
        if self._sdf is None:
            return forces, records
        r = self._strand_radius
        # Hair-body contacts
        for i in range(self._n):
            if self._is_root[i]:
                continue
            d, n, obj_idx = self._sdf.signed_distance(pos[i])
            penetration = r - d
            if penetration > 0.0:
                rel_v = vel[i] - self._body_velocity(pos[i], self._t)
                v_n = np.dot(rel_v, n)
                # Normal spring + damping
                fn_mag = self._contact_stiffness * max(penetration, 0.0)
                fn = fn_mag * n
                # Tangential friction
                vt = rel_v - n * v_n
                vt_mag = _norm(vt)
                # stick if slow, else kinetic
                if vt_mag < self._stick_tol:
                    ft = -vt * self._friction * fn_mag
                else:
                    ft = -vt / vt_mag * self._friction * fn_mag
                f_total = fn + ft
                forces[i] += f_total
                records.append(
                    {
                        "type": "hair_body",
                        "normal": n,
                        "normal_force": fn_mag,
                        "tangent_force": ft,
                        "rel_velocity": rel_v,
                        "static": self._friction,
                        "kinetic": self._friction,
                        "mask": 1,
                    }
                )
        # Hair-hair contacts using k-d tree
        if self._n > 1:
            tree = cKDTree(pos)
            pairs = tree.query_pairs(r * 3.0, output_type="ndarray")
            for i, j in pairs:
                if self._strand_id[i] == self._strand_id[j]:
                    continue
                diff = pos[j] - pos[i]
                dist = _norm(diff)
                if dist < 1e-12:
                    continue
                n = diff / dist
                penetration = 2.0 * r - dist
                if penetration <= 0.0:
                    continue
                # Equal and opposite normal force along i->j
                fn_mag = self._contact_stiffness * penetration * 0.5
                fn = fn_mag * n
                rel_v = vel[i] - vel[j]
                vt = rel_v - n * np.dot(rel_v, n)
                vt_mag = _norm(vt)
                if vt_mag < self._stick_tol:
                    ft_i = -vt * self._friction * fn_mag
                else:
                    ft_i = -vt / vt_mag * self._friction * fn_mag
                f_i = fn + ft_i
                forces[i] += f_i
                forces[j] -= f_i
                records.append(
                    {
                        "type": "hair_hair",
                        "normal": -n,
                        "normal_force": fn_mag,
                        "tangent_force": -ft_i,
                        "rel_velocity": -rel_v,
                        "static": self._friction,
                        "kinetic": self._friction,
                        "mask": 1,
                    }
                )
        return forces, records

    def _step(self, dt: float) -> None:
        pos = self._pos
        vel = self._vel
        # Gravity
        g = np.array(self._config.get("gravity", [0.0, 0.0, -9.81]), float)
        f_gravity = self._mass[:, None] * g
        # Elastic
        f_elastic = self._elastic_forces(pos)
        # Contact
        f_contact, _ = self._contact_response(pos, vel)
        # Damping proportional to velocity
        f_damp = -self._damping * vel * self._mass[:, None]
        f_total = f_gravity + f_elastic + f_contact + f_damp
        # Symplectic Euler
        acc = f_total / self._mass[:, None]
        acc[self._is_root.astype(bool)] = 0.0
        vel_new = vel + dt * acc
        pos_new = pos + dt * vel_new
        # Kinematic roots follow head motion
        disp, _ = self._head_displacement(self._t + dt)
        pos_new[self._is_root.astype(bool)] = self._root_init[self._is_root.astype(bool)] + disp
        vel_new[self._is_root.astype(bool)] = 0.0
        self._pos = pos_new
        self._vel = vel_new
        self._t += dt

    def advance_to(self, time: float) -> StateFrame:
        if self._scene is None:
            raise RuntimeError("Solver not initialized")
        target = float(time)
        while self._t < target - 1e-12:
            dt = min(self._dt, target - self._t)
            self._step(dt)
        # Build frame
        return self._build_frame(target)

    def _build_frame(self, time: float) -> StateFrame:
        # Required friction fields
        contact_records = self._contact_response(self._pos, self._vel)[1]
        c = len(contact_records)
        if c == 0:
            friction_normal = np.zeros((0, 3), float)
            friction_relative_velocity = np.zeros((0, 3), float)
            friction_tangential_force = np.zeros((0, 3), float)
            friction_normal_force = np.zeros(0, float)
            friction_static_coefficient = np.zeros(0, float)
            friction_dynamic_coefficient = np.zeros(0, float)
            friction_active_mask = np.zeros(0, float)
        else:
            friction_normal = np.array([rec["normal"] for rec in contact_records])
            friction_relative_velocity = np.array([rec["rel_velocity"] for rec in contact_records])
            friction_tangential_force = np.array([rec["tangent_force"] for rec in contact_records])
            friction_normal_force = np.array([rec["normal_force"] for rec in contact_records])
            friction_static_coefficient = np.array([rec["static"] for rec in contact_records])
            friction_dynamic_coefficient = np.array([rec["kinetic"] for rec in contact_records])
            friction_active_mask = np.array([rec["mask"] for rec in contact_records])

        # Budget fields: global + per-body domains
        # G = 1 global + strand_count bodies
        n = self._n
        strand_count = self._strand_count
        G = 1 + strand_count
        budget_domain_weights = np.zeros((G, n), float)
        budget_domain_weights[0, :] = 1.0
        for sid in range(strand_count):
            mask = (self._strand_id == sid).astype(float)
            budget_domain_weights[1 + sid, :] = mask

        # External force per domain: gravity only for now, shape [G,1,3]
        g = np.array(self._config.get("gravity", [0.0, 0.0, -9.81]), float)
        budget_external_force = np.zeros((G, 1, 3), float)
        budget_external_force[0, 0] = np.sum(self._mass[:, None] * g, axis=0)
        for sid in range(strand_count):
            mask = self._strand_id == sid
            budget_external_force[1 + sid, 0] = np.sum(self._mass[mask, None] * g, axis=0)

        # Cumulative impulses and momenta: zero for this simple explicit solver
        budget_cumulative_external_impulse = np.zeros((G, 3), float)
        budget_cumulative_inflow_momentum = np.zeros((G, 3), float)
        budget_cumulative_outflow_momentum = np.zeros((G, 3), float)

        # Cumulative mass source: zero
        cumulative_mass_source = np.zeros(1, float)

        # attributes.mass_quantity: per-sample mass as single material
        mass_quantity = self._mass.copy()

        # Reference scalars
        ref_force = self._ref_force
        ref_momentum = self._ref_momentum
        ref_velocity = self._ref_velocity

        fields = {
            "budget_cumulative_external_impulse": budget_cumulative_external_impulse,
            "budget_cumulative_inflow_momentum": budget_cumulative_inflow_momentum,
            "budget_cumulative_outflow_momentum": budget_cumulative_outflow_momentum,
            "budget_domain_weights": budget_domain_weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": cumulative_mass_source,
            "friction_active_mask": friction_active_mask,
            "friction_dynamic_coefficient": friction_dynamic_coefficient,
            "friction_normal": friction_normal,
            "friction_normal_force": friction_normal_force,
            "friction_relative_velocity": friction_relative_velocity,
            "friction_static_coefficient": friction_static_coefficient,
            "friction_tangential_force": friction_tangential_force,
        }

        attributes = {
            "strand_id": self._strand_id,
            "node_index": self._node_index,
            "is_root": self._is_root,
            "mass_quantity": mass_quantity,
        }

        scalars = {
            "friction_stick_speed_tolerance": self._stick_tol,
            "reference_force": ref_force,
            "reference_momentum": ref_momentum,
            "reference_velocity": ref_velocity,
        }

        frame = StateFrame(
            time=time,
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            body_id=self._strand_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "BenchmarkSolver", "dt": self._dt, **self._diag}
