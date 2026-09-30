"""Mass-spring viscoplastic bunny falling down a staircase.

Implements a simple explicit time-stepping physical solver.  The bunny is
coarsely represented by its input mesh vertices connected by the input
triangle/edge graph (faces loaded from the same NPZ asset).  Internal
elasticity is modelulated by edge springs plus a per-vertex volume penalty,
collision against the frozen boxes is handled by penalty forces, and
work-softening plasticity is approximated by reducing the effective stiffness
of vertices that have undergone large deformation rates/strain.

This is intentionally a minimal physically-based baseline: gravity, contact,
damping, softening and friction are all integrated by explicit forward Euler
steps with sub-stepping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._scene: Any | None = None
        self._seed: int = 0
        self._config: dict = {}
        self._rng = np.random.default_rng(0)

        # state
        self._time: float = 0.0
        self._x: NDArray[np.float64] | None = None
        self._v: NDArray[np.float64] | None = None
        self._mass: NDArray[np.float64] | None = None
        self._body_id: NDArray[np.int64] | None = None
        self._F: NDArray[np.float64] | None = None  # per-vert deformation gradient
        self._soft: NDArray[np.float64] | None = None  # per-vert softening factor [0,1]
        self._plastic_work: NDArray[np.float64] | None = None

        # mesh topology
        self._faces: NDArray[np.int64] | None = None
        self._edges: NDArray[np.int64] | None = None
        self._rest_lengths: NDArray[np.float64] | None = None
        self._rest_volume: float = 1.0

        # obstacles (boxes)
        self._boxes: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

        # sample times for which we have exact frames
        self._frame_cache: dict[float, StateFrame] = {}

    def initialize(self, scene: Any, seed: int) -> None:
        self._scene = scene
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._config = scene.config if hasattr(scene, "config") else {}

        init = scene.initial_state()
        self._time = float(init.time)
        self._x = np.asarray(init.position, dtype=float)
        self._v = np.asarray(init.velocity, dtype=float)
        self._mass = np.asarray(init.mass, dtype=float)
        self._body_id = np.asarray(init.body_id, dtype=np.int64) if init.body_id is not None else None
        n = len(self._x)

        self._F = np.asarray(init.attributes.get("deformation_gradient", np.tile(np.eye(3), (n, 1, 1))), dtype=float)
        self._soft = np.zeros(n, dtype=float)
        self._plastic_work = np.zeros(n, dtype=float)

        # Load topology from the same asset that the scene uses.
        asset_path = Path(self._config.get("subject_asset", "assets/subject.npz"))
        data = np.load(asset_path)
        if "faces" in data:
            self._faces = np.asarray(data["faces"], dtype=np.int64)
        else:
            # fallback: build a simple tet-like edge graph from nearest neighbours
            self._faces = None

        # Build unique edge list from faces (or k-neighbour fallback)
        self._edges = self._build_edges(n)
        x0 = self._x
        self._rest_lengths = np.linalg.norm(x0[self._edges[:, 1]] - x0[self._edges[:, 0]], axis=1)
        # rough rest volume for volume penalty
        if self._faces is not None:
            vols = []
            for a, b, c in self._faces:
                vols.append(np.linalg.norm(np.cross(self._x[b] - self._x[a], self._x[c] - self._x[a])) * 0.5)
            self._rest_volume = float(np.sum(vols)) if vols else 1.0
        else:
            self._rest_volume = 1.0

        # Collision boxes from scene_geometry.yaml
        self._boxes = []
        for obj in self._config.get("scene_geometry", {}).get("objects", []):
            if obj.get("collision_enabled") and obj["geometry"]["type"] == "box":
                t = np.asarray(obj["transform"]["translation"], dtype=float)
                h = np.asarray(obj["geometry"]["half_extents"], dtype=float)
                self._boxes.append((t - h, t + h))

        # Cache the initial frame
        self._frame_cache = {self._time: self._make_frame(init.fields.get("external_force", np.zeros((n, 3))))}

    def _build_edges(self, n: int) -> NDArray[np.int64]:
        if self._faces is not None and len(self._faces):
            edges = set()
            for f in self._faces:
                edges.add((int(min(f[0], f[1])), int(max(f[0], f[1]))))
                edges.add((int(min(f[1], f[2])), int(max(f[1], f[2]))))
                edges.add((int(min(f[2], f[0])), int(max(f[2], f[0]))))
            return np.asarray(sorted(edges), dtype=np.int64)
        # fallback: Delaunay-like nearest neighbours along z
        idx = np.argsort(self._x[:, 2])
        edges = []
        for i in range(1, len(idx)):
            a, b = idx[i - 1], idx[i]
            edges.append((int(min(a, b)), int(max(a, b))))
        return np.asarray(edges, dtype=np.int64)

    def _make_frame(self, ext_force: NDArray[np.float64] | None = None) -> StateFrame:
        x = self._x
        v = self._v
        n = len(x)
        # signed distance to nearest box surface (negative inside)
        gap = self._signed_distance(x)
        contact = gap[gap < 0.0] if len(gap) else np.array([], dtype=float)

        if ext_force is None:
            ext_force = np.zeros((n, 3), dtype=float)

        # interaction force is just the contact force per vertex aggregated by simple grouping
        interaction = self._contact_force(x, v).sum(axis=0).reshape(1, 3)

        attributes = {
            "mass_quantity": self._mass.copy(),
            "deformation_gradient": self._F.copy(),
        }
        fields = {
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "external_force": ext_force,
            "contact_signed_distance": contact,
            "interaction_force": interaction,
        }
        frame = StateFrame(
            time=self._time,
            position=x.copy(),
            velocity=v.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy() if self._body_id is not None else None,
            attributes=attributes,
            fields=fields,
            scalars={"contact_length_scale": float(self._config.get("contact_length_scale", 0.04))},
        )
        frame.validate()
        return frame

    def _signed_distance(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # For each vertex, signed distance to union of boxes (negative inside).
        sd = np.full(len(x), 1e6)
        for lo, hi in self._boxes:
            d = np.maximum(lo - x, x - hi)  # negative inside
            outside = np.any(d > 0, axis=1)
            abs_dist = np.abs(d)
            # outside: positive distance to closest face; inside: negative distance
            dist_out = np.where(outside, abs_dist.min(axis=1), -abs_dist.min(axis=1))
            sd = np.minimum(sd, dist_out)
        return sd

    def _contact_force(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        n = len(x)
        f = np.zeros((n, 3), dtype=float)
        h = float(self._config.get("contact_length_scale", 0.04))
        k = float(self._config.get("stiffness", 22.0))
        restitution = float(self._config.get("restitution", 0.12))
        friction = float(self._config.get("friction", 0.72))
        for lo, hi in self._boxes:
            d = np.maximum(lo - x, x - hi)  # negative inside, positive outside
            inside = np.all(d < 0, axis=1)
            if not np.any(inside):
                continue
            # penetration depth = - signed distance for inside points
            dist = -np.max(d, axis=1)
            dist[~inside] = 0.0
            # normal points outward from box (largest face-axis penetration)
            absd = np.abs(d)
            axis = np.argmax(absd, axis=1)
            normal = np.zeros((n, 3), dtype=float)
            rows = np.arange(n)
            normal[rows, axis] = np.sign(d[rows, axis] + 1e-12)
            normal[~inside] = 0.0
            # penalty force + damping
            vn = np.sum(v * normal, axis=1, keepdims=True)
            fn = (-k / h * dist[:, None] - restitution * np.clip(vn, -1e3, 1e3)) * normal
            # friction opposes tangential velocity
            vt = v - vn * normal
            vt_mag = np.linalg.norm(vt, axis=1, keepdims=True)
            vt_dir = vt / (vt_mag + 1e-9)
            friction_force = -friction * np.abs(fn) * vt_dir
            friction_force[vt_mag[:, 0] < 1e-6] = 0.0
            f += fn + friction_force
        return f

    def _elastic_force(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        n = len(x)
        f = np.zeros((n, 3), dtype=float)
        k = float(self._config.get("stiffness", 22.0))
        # apply softening
        if self._soft is not None:
            soft = 1.0 - np.clip(self._soft, 0.0, 0.95)
        else:
            soft = np.ones(n, dtype=float)
        e0, e1 = self._edges[:, 0], self._edges[:, 1]
        dx = x[e1] - x[e0]
        l = np.linalg.norm(dx, axis=1)
        l0 = self._rest_lengths
        strain = (l - l0) / (l0 + 1e-12)
        # force along edge
        dir_ = dx / (l[:, None] + 1e-12)
        # average softening of two endpoints
        s = 0.5 * (soft[e0] + soft[e1])
        mag = k * s * strain
        f_e = dir_ * mag[:, None]
        np.add.at(f, e0, f_e)
        np.add.at(f, e1, -f_e)
        return f

    def _volume_force(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # simple centroid pressure preserving the current mean volume
        n = len(x)
        f = np.zeros((n, 3), dtype=float)
        if self._faces is None or len(self._faces) == 0:
            return f
        k_vol = float(self._config.get("stiffness", 22.0)) * 0.1
        # compute current volume
        a, b, c = self._faces[:, 0], self._faces[:, 1], self._faces[:, 2]
        vols = np.linalg.norm(np.cross(x[b] - x[a], x[c] - x[a]), axis=1) * 0.5
        current = float(vols.sum())
        ratio = current / (self._rest_volume + 1e-12)
        if abs(ratio - 1.0) < 1e-3:
            return f
        # apply outward pressure proportional to volume change
        for i, (i0, i1, i2) in enumerate(self._faces):
            normal = np.cross(x[i1] - x[i0], x[i2] - x[i0])
            area = np.linalg.norm(normal) + 1e-12
            normal = normal / area
            press = -k_vol * (ratio - 1.0) * area
            f[i0] += press * normal
            f[i1] += press * normal
            f[i2] += press * normal
        return f

    def _update_deformation(self, dt: float) -> None:
        # Approximate deformation gradient evolution using edge/face based changes.
        # For simplicity, F is updated as a rotation-free scaling driven by the
        # velocity gradient around each vertex, and plastic work softens regions
        # with large strain rates.
        n = len(self._x)
        for i in range(n):
            # estimate velocity gradient from edges
            L = np.zeros((3, 3))
            for j in range(len(self._edges)):
                a, b = self._edges[j]
                if a == i or b == i:
                    other = b if a == i else a
                    l = np.linalg.norm(self._x[other] - self._x[i]) + 1e-12
                    dv = self._v[other] - self._v[i]
                    dx = (self._x[other] - self._x[i]) / l
                    L += np.outer(dv, dx) / (l + 1e-12)
            L /= max(len(self._edges) / max(n, 1), 1.0)
            # Fdot = L F
            self._F[i] += dt * (L @ self._F[i])
            # plastic softening
            strain_rate = np.linalg.norm(0.5 * (L + L.T))
            self._plastic_work[i] += strain_rate * float(self._config.get("softening_rate", 0.34)) * dt
            self._soft[i] = min(self._soft[i] + strain_rate * dt, 0.95)

    def _step(self, dt: float, ext_force: NDArray[np.float64]) -> None:
        x = self._x
        v = self._v
        n = len(x)
        mass = self._mass[:, None]
        g = np.asarray(self._config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # forces
        f_elastic = self._elastic_force(x)
        f_volume = self._volume_force(x)
        f_contact = self._contact_force(x, v)
        f_damping = -float(self._config.get("damping", 2.3)) * v

        # external_force field passed from the scene already includes gravity.
        f_total = f_elastic + f_volume + f_contact + f_damping + ext_force

        # explicit integration with substep damping
        v_new = v + dt * f_total / mass
        x_new = x + dt * v_new

        self._v = v_new
        self._x = x_new
        self._time += dt
        self._update_deformation(dt)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        # If already cached, return it.
        if target in self._frame_cache:
            return self._frame_cache[target]

        # Determine the previous sample time and step from there.
        # The runner advances monotonically increasing time.
        sample_dt = float(self._config.get("sample_dt", 0.03333333333333333))
        n_sub = max(1, int(np.ceil(sample_dt / 0.001)))
        sub_dt = sample_dt / n_sub

        ext_force = self._frame_cache.get(self._time, None)
        if ext_force is not None:
            ext_force = ext_force.fields.get("external_force", np.zeros((len(self._x), 3)))
        else:
            ext_force = np.zeros((len(self._x), 3), dtype=float)

        while self._time < target - 1e-12:
            dt = min(sub_dt, target - self._time)
            self._step(dt, ext_force)

        frame = self._make_frame(ext_force)
        self._frame_cache[target] = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"seed": self._seed, "solver": "viscoplastic_mass_spring"}
