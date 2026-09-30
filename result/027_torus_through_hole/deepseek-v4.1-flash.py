"""Physics solver for the deformable torus pushed through a rigid hole.

Physical model
--------------
The subject is a closed triangular mesh (a torus) discretised as a mass-spring
shell.  Every mesh edge carries a stiff spring whose rest length comes from the
frozen asset, and second-ring springs (graph distance two) provide the bending
and shear stiffness that keeps the tube and ring from crumpling.  Each spring
also carries a viscous dashpot.  The only volume load is gravity; the rigid
plate, the ground and the kinematically driven pusher are enforced as
non-penetration constraints (position projection onto the obstacle surface)
combined with Coulomb friction and a small restitution.

Time integration is a fixed-step symplectic (semi-implicit) Euler scheme whose
step resolves the stiffest spring of the network, so the simulation is
deterministic, stable and finite at every sample time.  Sampled frames report
the per-vertex deformation gradient, per-vertex mass, the external (gravity)
force, the per-vertex signed gap to the nearest obstacle and the contact
normalisation length.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # pragma: no cover - platform contract
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = None


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of 2D points, returned counter clockwise."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def _edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]], axis=0
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _second_ring_pairs(edges: np.ndarray, n: int) -> np.ndarray:
    adj: list = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    pairs = set()
    for i in range(n):
        first = set(adj[i])
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k not in first:
                    pairs.add((i, k) if i < k else (k, i))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def _vertex_normals(x: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = x[faces[:, 0]]
    p1 = x[faces[:, 1]]
    p2 = x[faces[:, 2]]
    face_n = np.cross(p1 - p0, p2 - p0)
    n = len(x)
    out = np.zeros_like(x)
    for c in range(3):
        for d in range(3):
            out[:, d] += np.bincount(
                faces[:, c], weights=face_n[:, d], minlength=n
            )
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(length, 1e-12)


def _load_yaml(path: Path):
    try:
        from worldsolver.config import load_yaml

        return load_yaml(path)
    except Exception:
        return None


class BenchmarkSolver:
    """Deterministic mass-spring torus driven through a rigid perforated plate."""

    #: spring stiffness per unit *stiffness* config value (N/m)
    _EDGE_STIFFNESS_SCALE = 20.0
    #: second-ring (bending/shear) springs relative to edge springs
    _SECOND_RING_SCALE = 0.5
    #: dashpot coefficient relative to critical damping of one spring
    _DASHPOT_RATIO = 0.10
    #: mild global velocity damping (1/s) per unit *damping* config value
    _GLOBAL_DAMPING = 0.10
    #: fixed integration step (s); resolves the stiffest spring with margin
    _DT = 2.5e-4

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        self.seed = int(seed)
        base = Path(__file__).resolve().parents[1]

        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.n = int(self.x.shape[0])
        self.gravity = np.asarray(
            cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float
        ).reshape(3)

        asset = cfg.get("subject_asset", "assets/subject.npz")
        mesh_path = Path(asset)
        if not mesh_path.is_absolute():
            mesh_path = base / mesh_path
        with np.load(mesh_path) as data:
            faces = np.asarray(data["faces"], dtype=np.int64)
            rest = np.asarray(data["vertices"], dtype=float)
        self.faces = faces
        if rest.shape != self.x.shape:
            rest = self.x.copy()
        self.rest = rest

        # --- spring network ------------------------------------------- #
        edges = _edges_from_faces(faces)
        second = _second_ring_pairs(edges, self.n)
        self.si = np.concatenate([edges[:, 0], second[:, 0]]).astype(np.int64)
        self.sj = np.concatenate([edges[:, 1], second[:, 1]]).astype(np.int64)
        stiff_cfg = float(cfg.get("stiffness", 11.0))
        k_edge = self._EDGE_STIFFNESS_SCALE * stiff_cfg
        k_second = self._SECOND_RING_SCALE * k_edge
        self.k = np.concatenate(
            [np.full(len(edges), k_edge), np.full(len(second), k_second)]
        )
        self.length0 = np.linalg.norm(
            self.rest[self.si] - self.rest[self.sj], axis=1
        )
        mean_mass = float(np.mean(self.mass))
        damp_cfg = float(cfg.get("damping", 1.6))
        self.c_damp = (
            self._DASHPOT_RATIO
            * damp_cfg
            * np.sqrt(self.k * max(mean_mass, 1e-12))
        )
        self.alpha = self._GLOBAL_DAMPING * damp_cfg

        # --- deformation-gradient operator (1-ring + virtual normal) -- #
        adj = self._one_ring(edges)
        self.nbr = adj
        delta = float(np.mean(self.length0))
        self.delta = max(delta, 1e-9)
        n0 = _vertex_normals(self.rest, faces)
        rest_dx = np.concatenate(
            [
                self.rest[adj] - self.rest[:, None, :],
                (self.delta * n0)[:, None, :],
            ],
            axis=1,
        )
        self._F_operator = np.linalg.pinv(rest_dx.transpose(0, 2, 1))

        # --- rigid obstacles ------------------------------------------ #
        self._init_obstacles(cfg, base)

        # --- integrator state ----------------------------------------- #
        self.t = 0.0
        self.steps = 0
        self._ready = True

    @staticmethod
    def _one_ring(edges: np.ndarray) -> np.ndarray:
        n = int(max(edges.max(), 0)) + 1
        adj: list = [[] for _ in range(n)]
        for a, b in edges:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
        width = max(len(a) for a in adj)
        out = np.full((n, width), -1, dtype=np.int64)
        for i, a in enumerate(adj):
            out[i, : len(a)] = a
        return np.where(out < 0, np.arange(n, dtype=np.int64)[:, None], out)

    # ------------------------------------------------------------------ #
    # rigid obstacles
    # ------------------------------------------------------------------ #
    def _init_obstacles(self, cfg: dict, base: Path) -> None:
        geo = _load_yaml(base / "scene_geometry.yaml")
        if not isinstance(geo, dict):
            geo = {}
        objects = {
            str(o.get("id")): o
            for o in geo.get("objects", [])
            if isinstance(o, dict)
        }
        pusher = objects.get("pusher", {})
        pgeo = pusher.get("geometry", {}) or {}
        self.pusher_radius = float(pgeo.get("radius", 0.48))
        self.pusher_half_height = 0.5 * float(pgeo.get("height", 0.18))
        ptrans = (pusher.get("transform", {}) or {}).get("translation", [0.0, 0.0, 1.88])
        self.pusher_z0 = float(ptrans[2]) if len(ptrans) >= 3 else 1.88

        self.hole_z = float(cfg.get("hole_z", 0.62))
        self.hole_radius = float(cfg.get("hole_radius", 0.5))
        self.push_speed = float(cfg.get("push_speed", 0.48))
        # the pusher stops once its lower face reaches the hole mid plane, so it
        # can never drive the subject into the ground nor cut into the plate.
        self.pusher_z_stop = self.hole_z - 0.02 + self.pusher_half_height

        poly = None
        mesh = (objects.get("rigid_hole", {}) or {}).get("geometry", {}) or {}
        obj_path = mesh.get("asset")
        if obj_path:
            candidate = Path(str(obj_path))
            if not candidate.is_absolute():
                candidate = base / candidate
            poly = self._plate_profile(candidate)
        if poly is None:
            poly = np.asarray(
                [
                    [self.hole_radius, self.hole_z],
                    [0.73, self.hole_z - 0.23],
                    [0.96, self.hole_z],
                    [0.73, self.hole_z + 0.23],
                ],
                dtype=float,
            )
        self.plate_poly = _convex_hull_2d(poly)

        self.friction = float(cfg.get("friction", 0.48))
        self.restitution = float(cfg.get("restitution", 0.04))
        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.04))
        self.contact_vel_margin = max(0.25 * self.contact_length_scale, 1e-3)
        self._pusher_z = self.pusher_z0

    @staticmethod
    def _plate_profile(path: Path, max_points: int = 4000):
        try:
            pts = []
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if not line.startswith("v "):
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    pts.append((np.hypot(x, y), z))
                    if len(pts) >= max_points:
                        break
        except Exception:
            return None
        if len(pts) < 3:
            return None

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        p = sorted(set(pts))
        lower: list = []
        for q in p:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0.0:
                lower.pop()
            lower.append(q)
        upper: list = []
        for q in reversed(p):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0.0:
                upper.pop()
            upper.append(q)
        return np.asarray(lower[:-1] + upper[:-1], dtype=float)

    def _pusher_position(self, t: float) -> float:
        z = self.pusher_z0 - self.push_speed * t
        return max(z, self.pusher_z_stop)

    def _pusher_velocity(self, t: float) -> float:
        return -self.push_speed if self._pusher_z > self.pusher_z_stop + 1e-12 else 0.0

    # ------------------------------------------------------------------ #
    # obstacle signed distances
    # ------------------------------------------------------------------ #
    def _ground_sdf(self, x):
        d = x[:, 2].copy()
        n = np.zeros_like(x)
        n[:, 2] = 1.0
        return d, n, np.zeros(3)

    def _plate_sdf(self, x):
        rho = np.hypot(x[:, 0], x[:, 1])
        p = np.stack([rho, x[:, 2]], axis=1)
        a = self.plate_poly
        b = np.roll(a, -1, axis=0)
        e = b - a
        w = p[:, None, :] - a[None, :, :]
        denom = np.maximum((e * e).sum(axis=1), 1e-18)
        t = np.clip((w * e[None, :, :]).sum(axis=-1) / denom[None, :], 0.0, 1.0)
        proj = a[None, :, :] + t[..., None] * e[None, :, :]
        to_boundary = proj - p[:, None, :]
        dist = np.linalg.norm(to_boundary, axis=-1)
        cross = e[None, :, 0] * w[..., 1] - e[None, :, 1] * w[..., 0]
        inside = np.all(cross >= -1e-12, axis=1)
        idx = np.arange(p.shape[0])
        k = np.argmin(dist, axis=1)
        dmin = dist[idx, k]
        vec = to_boundary[idx, k]
        length = np.maximum(np.linalg.norm(vec, axis=1), 1e-12)
        sgn = np.where(inside, -1.0, 1.0)
        d = sgn * dmin
        # outward unit normal in the (rho, z) half plane: points away from the
        # solid, i.e. opposite to (point -> boundary) for outside points and
        # along (point -> boundary) for inside points.
        n2 = (-sgn)[:, None] * vec / length[:, None]
        rho_safe = np.where(rho > 1e-12, rho, 1.0)
        er_x = np.where(rho > 1e-12, x[:, 0] / rho_safe, 1.0)
        er_y = np.where(rho > 1e-12, x[:, 1] / rho_safe, 0.0)
        n = np.stack([n2[:, 0] * er_x, n2[:, 0] * er_y, n2[:, 1]], axis=1)
        return d, n, np.zeros(3)

    def _pusher_sdf(self, x, vz):
        zc = self._pusher_z
        rho = np.hypot(x[:, 0], x[:, 1])
        rho_safe = np.where(rho > 1e-12, rho, 1.0)
        er_x = np.where(rho > 1e-12, x[:, 0] / rho_safe, 1.0)
        er_y = np.where(rho > 1e-12, x[:, 1] / rho_safe, 0.0)
        q0 = rho - self.pusher_radius
        q1 = np.abs(x[:, 2] - zc) - self.pusher_half_height
        out = (q0 > 0.0) | (q1 > 0.0)
        d = np.where(
            out,
            np.hypot(np.maximum(q0, 0.0), np.maximum(q1, 0.0)),
            np.maximum(q0, q1),
        )
        sz = np.where(x[:, 2] >= zc, 1.0, -1.0)
        g = np.zeros_like(x)
        gr = np.where(q0 > 0.0, q0, 0.0)
        gz = np.where(q1 > 0.0, q1, 0.0)
        g[:, 0] = gr * er_x
        g[:, 1] = gr * er_y
        g[:, 2] = gz * sz
        glen = np.linalg.norm(g, axis=1)
        good = glen > 1e-12
        n = np.zeros_like(x)
        n[good] = g[good] / glen[good, None]
        deep_radial = (~out) & (q0 >= q1)
        n[deep_radial, 0] = er_x[deep_radial]
        n[deep_radial, 1] = er_y[deep_radial]
        n[deep_radial, 2] = 0.0
        deep_z = (~out) & (q1 > q0)
        n[deep_z, :] = 0.0
        n[deep_z, 2] = sz[deep_z]
        return d, n, np.asarray([0.0, 0.0, vz], dtype=float)

    def _obstacles(self, t: float):
        self._pusher_z = self._pusher_position(t)
        vz = self._pusher_velocity(t)
        return (
            self._ground_sdf,
            self._plate_sdf,
            lambda pts: self._pusher_sdf(pts, vz),
        )

    # ------------------------------------------------------------------ #
    # contact and time stepping
    # ------------------------------------------------------------------ #
    def _resolve_contacts(self, t: float) -> None:
        for _ in range(2):
            for sdf in self._obstacles(t):
                d, n, v_obs = sdf(self.x)
                pen = d < 0.0
                if np.any(pen):
                    self.x[pen] -= d[pen, None] * n[pen]
                active = d < self.contact_vel_margin
                if not np.any(active):
                    continue
                rel = self.v[active] - v_obs[None, :]
                vn = np.einsum("ij,ij->i", rel, n[active])
                dvn = np.where(vn < 0.0, -(1.0 + self.restitution) * vn, 0.0)
                self.v[active] += dvn[:, None] * n[active]
                vt = rel - vn[:, None] * n[active]
                vtm = np.linalg.norm(vt, axis=1)
                slide = np.minimum(self.friction * np.abs(dvn), vtm)
                moving = vtm > 1e-12
                corr = np.zeros_like(vt)
                corr[moving] = (
                    slide[moving, None] * vt[moving] / vtm[moving, None]
                )
                self.v[active] -= corr

    def _step(self, dt: float, t: float) -> None:
        dx = self.x[self.si] - self.x[self.sj]
        length = np.linalg.norm(dx, axis=1)
        safe = np.maximum(length, 1e-12)
        direction = dx / safe[:, None]
        force = self.k * (length - self.length0)
        rel_v = np.einsum(
            "ij,ij->i", self.v[self.si] - self.v[self.sj], direction
        )
        force = force + self.c_damp * rel_v
        fvec = force[:, None] * direction
        acc = np.zeros_like(self.x)
        for axis in range(3):
            acc[:, axis] = np.bincount(
                self.si, weights=fvec[:, axis], minlength=self.n
            ) - np.bincount(self.sj, weights=fvec[:, axis], minlength=self.n)
        acc /= self.mass[:, None]
        acc += self.gravity[None, :]
        self.v += dt * acc
        self.v *= np.exp(-self.alpha * dt)
        self.x += dt * self.v
        self._resolve_contacts(t + dt)
        self.t = t + dt
        self.steps += 1

    # ------------------------------------------------------------------ #
    # frame assembly
    # ------------------------------------------------------------------ #
    def _all_contact_distances(self, t: float) -> np.ndarray:
        best = None
        for sdf in self._obstacles(t):
            d = sdf(self.x)[0]
            best = d if best is None else np.minimum(best, d)
        return best

    def _deformation_gradient(self) -> np.ndarray:
        normals = _vertex_normals(self.x, self.faces)
        dx = np.concatenate(
            [
                self.x[self.nbr] - self.x[:, None, :],
                (self.delta * normals)[:, None, :],
            ],
            axis=1,
        )
        op = self._F_operator
        grad = np.einsum("nij,njk->nik", dx.transpose(0, 2, 1), op)
        bad = ~np.isfinite(grad)
        if np.any(bad):
            grad = np.where(bad, np.eye(3)[None, :, :], grad)
        return grad

    def _make_frame(self, time: float):
        grad = self._deformation_gradient()
        distances = self._all_contact_distances(time)
        fields = {
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "external_force": self.mass[:, None] * self.gravity[None, :],
            "contact_signed_distance": distances,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "deformation_gradient": grad,
        }
        scalars = {"contact_length_scale": float(self.contact_length_scale)}
        if StateFrame is None:  # pragma: no cover
            return dict(
                time=time,
                position=self.x.copy(),
                velocity=self.v.copy(),
                mass=self.mass.copy(),
                attributes=attributes,
                fields=fields,
                scalars=scalars,
            )
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() has not been called")
        guard = 0
        while self.t < target - 1e-12:
            dt = min(self._DT, target - self.t)
            self._step(dt, self.t)
            guard += 1
            if guard > 5_000_000:
                break
        return self._make_frame(target)

    def diagnostics(self) -> dict:
        grad = self._deformation_gradient()
        det = np.linalg.det(grad)
        return {
            "steps": int(self.steps),
            "dt": float(self._DT),
            "simulation_time": float(self.t),
            "min_contact_distance": float(
                np.min(self._all_contact_distances(self.t))
            ),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))),
            "min_deformation_det": float(np.min(det)),
            "max_deformation_det": float(np.max(det)),
            "contact_length_scale": float(self.contact_length_scale),
        }
