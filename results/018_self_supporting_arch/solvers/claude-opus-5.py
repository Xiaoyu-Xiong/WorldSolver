"""Self-supporting parabolic arch: rigid multi-body dynamics with friction.

Physical model
--------------
The fifteen voussoir blocks are rigid bodies carrying the frozen scene masses
and mesh-integrated inertia tensors.  Each substep advances the system with a
semi-implicit (symplectic) Euler scheme:

  1. free velocity update from the external force field (gravity only),
  2. geometric contact detection against the frozen meshes and the static
     ground plane z = 0,
  3. velocity-level solution of the discrete Signorini-Coulomb complementarity
     problem by warm-started projected Gauss-Seidel relaxation (sequential
     impulses),

        0 <= lambda_n  _|_  (n . v_rel + b) >= 0 ,
        |lambda_t| <= mu * lambda_n   with maximum dissipation,

  4. position / quaternion integration with the post-contact velocities.

Contacts are generated from geometry every substep: supporting-face corner
points of each voussoir against the mating face plane of its neighbour, and
mesh vertices against the ground plane.  The compressive load path of the arch
emerges purely from the contact solve; no motion is scripted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# Covariance of the canonical tetrahedron (0, e0, e1, e2), used for the exact
# mesh inertia integration by divergence-theorem tetrahedral decomposition.
_CANON_COV = np.array(
    [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float
) / 120.0


def _load_obj(path: Path):
    """Read a triangle-soup OBJ, returning (vertices[V,3], faces[F,3])."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def _mesh_inertia(vertices, faces, mass):
    """Exact volume, centroid and inertia tensor of a closed triangle mesh."""
    volume = 0.0
    com = np.zeros(3)
    cov = np.zeros((3, 3))
    for tri in faces:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        jac = np.column_stack((a, b, c))
        det = float(np.linalg.det(jac))
        volume += det / 6.0
        com += det / 24.0 * (a + b + c)
        cov += det * (jac @ _CANON_COV @ jac.T)
    if volume < 0.0:  # inward winding: flip all signed contributions
        volume, com, cov = -volume, -com, -cov
    volume = max(volume, 1e-12)
    com = com / volume
    # Shift covariance to the centre of mass, then convert to inertia.
    cov = cov / volume - np.outer(com, com)
    inertia = mass * (np.trace(cov) * np.eye(3) - cov)
    inertia = 0.5 * (inertia + inertia.T)
    # Guard against a degenerate mesh so the inverse always exists.
    eigmin = float(np.min(np.linalg.eigvalsh(inertia)))
    if not np.isfinite(eigmin) or eigmin <= 1e-9 * mass:
        inertia = inertia + np.eye(3) * (1e-6 * mass)
    return volume, com, inertia


def _quat_to_matrix(q):
    """Rotation matrix of a unit quaternion given in xyzw order."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n <= 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=float,
    )


def _integrate_quat(q, omega, dt):
    """Exponential-free quaternion update q <- normalize(q + dt/2 * omega x q)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])
    h = 0.5 * dt
    nx = x + h * (w * wx + wy * z - wz * y)
    ny = y + h * (w * wy + wz * x - wx * z)
    nz = z + h * (w * wz + wx * y - wy * x)
    nw = w - h * (wx * x + wy * y + wz * z)
    norm = (nx * nx + ny * ny + nz * nz + nw * nw) ** 0.5
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.array([nx / norm, ny / norm, nz / norm, nw / norm])


def _support_face(vertices, faces, direction, tol=1e-6):
    """Corner points of the mesh face whose outward normal best matches ``direction``.

    Returns (unit_normal, corner_points) in local body coordinates.  Corners are
    the mesh vertices lying on the supporting plane, ordered deterministically.
    """
    d = np.asarray(direction, dtype=float)
    nd = float(np.linalg.norm(d))
    d = d / nd if nd > 1e-12 else np.array([1.0, 0.0, 0.0])
    best_normal = d.copy()
    best_score = -np.inf
    best_ref = vertices[0]
    for tri in faces:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        nrm = np.cross(b - a, c - a)
        ln = float(np.linalg.norm(nrm))
        if ln < 1e-12:
            continue
        nrm = nrm / ln
        score = float(nrm @ d)
        if score > best_score:
            best_score, best_normal, best_ref = score, nrm, a
    offset = float(best_normal @ best_ref)
    dist = vertices @ best_normal - offset
    on_plane = np.nonzero(np.abs(dist) <= max(tol, 1e-5))[0]
    if on_plane.size == 0:
        on_plane = np.array([int(np.argmax(vertices @ best_normal))])
    corners = vertices[on_plane]
    # Deterministic ordering (lexicographic) so contact ids are frame stable.
    order = np.lexsort((corners[:, 2], corners[:, 1], corners[:, 0]))
    return best_normal, corners[order]


class BenchmarkSolver:
    """Warm-started sequential-impulse rigid body solver with Coulomb friction."""

    SUBSTEP_DT = 1.5e-3     # physics substep (s)
    PGS_ITERS = 20          # projected Gauss-Seidel sweeps per substep
    CONTACT_MARGIN = 8.0e-3  # speculative contact detection band (m)
    TOUCH_BAND = 2.0e-3     # gap below which Coulomb friction is applied (m)
    SLOP = 1.0e-4           # allowed penetration before positional bias (m)
    ERP = 0.25              # Baumgarte error-reduction factor
    MAX_BIAS_VEL = 0.5      # cap on penetration-recovery velocity (m/s)

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(getattr(scene, "config", {}) or {})
        self.gravity = np.asarray(
            cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)
        fric = cfg.get("friction", {})
        if isinstance(fric, dict):
            self.mu = float(fric.get("coefficient", 0.6))
        else:
            self.mu = float(fric)
        self.duration = float(cfg.get("duration", 3.0))
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.n = int(len(frame0.position))
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.zeros((self.n, 3), dtype=float)
        if frame0.velocity is not None:
            self.v = np.array(frame0.velocity, dtype=float)
        self.omega = np.zeros((self.n, 3), dtype=float)
        if frame0.angular_velocity is not None:
            self.omega = np.array(frame0.angular_velocity, dtype=float)
        if frame0.orientation is not None:
            self.q = np.array(frame0.orientation, dtype=float)
        else:
            self.q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1))
        self.mass = (
            np.array(frame0.mass, dtype=float)
            if frame0.mass is not None
            else np.ones(self.n)
        )
        self.body_id = (
            np.array(frame0.body_id, dtype=int)
            if frame0.body_id is not None
            else np.arange(self.n, dtype=int)
        )
        self.inv_mass = 1.0 / np.maximum(self.mass, 1e-12)

        self._load_geometry()
        self._build_contact_spec()

        self.x0 = self.x.copy()
        self.time = 0.0
        self.steps = 0
        self.max_penetration = 0.0
        self._warm: dict[tuple, tuple[float, float, float]] = {}
        self._R = np.stack([_quat_to_matrix(self.q[i]) for i in range(self.n)])
        self._update_world_inertia()

    def _load_geometry(self) -> None:
        root = Path(__file__).resolve().parent.parent
        assets = sorted((root / "assets").glob("voussoir_*.obj"))
        self.verts_local: list[np.ndarray] = []
        self.faces: list[np.ndarray] = []
        self.inertia: list[np.ndarray] = []
        self.inv_inertia_body: list[np.ndarray] = []
        sizes = []
        for i in range(self.n):
            if i < len(assets):
                verts, faces = _load_obj(assets[i])
            else:  # defensive fallback: unit cube of the same mass
                verts = np.array(
                    [
                        [sx * 0.25, sy * 0.4, sz * 0.25]
                        for sx in (-1, 1)
                        for sy in (-1, 1)
                        for sz in (-1, 1)
                    ],
                    dtype=float,
                )
                faces = np.array(
                    [
                        [0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7],
                        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
                    ],
                    dtype=int,
                )
            _, com, inertia = _mesh_inertia(verts, faces, float(self.mass[i]))
            # Body frame origin is the mesh origin used by the frozen scene
            # transforms; recentre only the inertia about the true centroid.
            self.verts_local.append(verts)
            self.faces.append(faces)
            self.inertia.append(inertia)
            self.inv_inertia_body.append(np.linalg.inv(inertia))
            ext = verts.max(axis=0) - verts.min(axis=0)
            sizes.append(float(np.mean(ext)))
        self.block_size = float(np.mean(sizes)) if sizes else 1.0

    def _build_contact_spec(self) -> None:
        """Freeze mating-face pairs and ground contact samples from t = 0 geometry."""
        self.pairs = []
        for a in range(self.n - 1):
            b = a + 1
            d_ab = self.x[b] - self.x[a]
            n_a, corners_a = _support_face(self.verts_local[a], self.faces[a], d_ab)
            n_b, corners_b = _support_face(self.verts_local[b], self.faces[b], -d_ab)
            self.pairs.append(
                {
                    "a": a,
                    "b": b,
                    "corners_a": corners_a,
                    "normal_b": n_b,
                    "ref_b": corners_b[0].copy(),
                }
            )
        # Frozen ground samples: the lowest vertex of every block at t = 0.
        self.ground_samples = []
        for i in range(self.n):
            world_z = self.verts_local[i][:, 2] + self.x[i][2]
            self.ground_samples.append((i, int(np.argmin(world_z))))
        self.n_gaps = 4 * len(self.pairs) + len(self.ground_samples)
        self.contact_length_scale = float(self.block_size)
        self.constraint_scale = float(self.block_size)

    def _update_world_inertia(self) -> None:
        self._Iinv = np.stack(
            [
                self._R[i] @ self.inv_inertia_body[i] @ self._R[i].T
                for i in range(self.n)
            ]
        )

    # ------------------------------------------------------------- kinematics
    def _world_vertices(self, i):
        return self.verts_local[i] @ self._R[i].T + self.x[i]

    # ------------------------------------------------------- contact assembly
    def _detect_contacts(self):
        """Geometric contact detection -> list of (bodyA, bodyB, point, normal, gap).

        ``bodyB == -1`` denotes the static ground plane.  ``normal`` points from
        B towards A, so a positive normal impulse pushes A away from B.
        """
        contacts = []
        margin = self.CONTACT_MARGIN
        world = [self._world_vertices(i) for i in range(self.n)]
        # Ground plane z = 0 (static frictional boundary).
        for i in range(self.n):
            pts = world[i]
            below = np.nonzero(pts[:, 2] < margin)[0]
            for k in below:
                p = pts[k]
                contacts.append(
                    (i, -1, p, np.array([0.0, 0.0, 1.0]), float(p[2]), ("g", i, int(k)))
                )
        # Mating voussoir faces.
        for pid, pair in enumerate(self.pairs):
            a, b = pair["a"], pair["b"]
            nb = self._R[b] @ pair["normal_b"]
            ref = self._R[b] @ pair["ref_b"] + self.x[b]
            pts_a = pair["corners_a"] @ self._R[a].T + self.x[a]
            for k in range(pts_a.shape[0]):
                gap = float(nb @ (pts_a[k] - ref))
                if gap < margin:
                    point = pts_a[k] - 0.5 * gap * nb
                    contacts.append((a, b, point, nb, gap, ("p", pid, k)))
        return contacts

    def _substep(self, dt: float) -> None:
        # 1. free velocity update (external force field = gravity).
        self.v += dt * self.gravity

        contacts = self._detect_contacts()
        if contacts:
            self._solve_contacts(contacts, dt)

        # 4. integrate configuration with post-contact velocities.
        self.x += dt * self.v
        for i in range(self.n):
            self.q[i] = _integrate_quat(self.q[i], self.omega[i], dt)
            self._R[i] = _quat_to_matrix(self.q[i])
        self._update_world_inertia()
        self.steps += 1

    def _solve_contacts(self, contacts, dt: float) -> None:
        """Warm-started projected Gauss-Seidel on the frictional contact LCP."""
        v = self.v
        om = self.omega
        invm = self.inv_mass
        Iinv = self._Iinv
        mu = self.mu
        rows = []       # per contact: geometry, effective masses, bias, limits
        warm_new: dict[tuple, tuple[float, float, float]] = {}
        for (a, b, point, n, gap, key) in contacts:
            # Deterministic tangent basis from the normal.
            ax = int(np.argmin(np.abs(n)))
            helper = np.zeros(3)
            helper[ax] = 1.0
            t1 = np.cross(n, helper)
            ln = float(np.linalg.norm(t1))
            if ln < 1e-9:
                t1 = np.array([1.0, 0.0, 0.0])
            else:
                t1 = t1 / ln
            t2 = np.cross(n, t1)
            ra = point - self.x[a]
            rb = point - self.x[b] if b >= 0 else np.zeros(3)
            dirs = (n, t1, t2)
            row = {"a": a, "b": b, "ra": ra, "rb": rb, "dirs": dirs}
            meff = []
            for d in dirs:
                k = invm[a] + float((np.cross(ra, d)) @ (Iinv[a] @ np.cross(ra, d)))
                if b >= 0:
                    k += invm[b] + float(
                        (np.cross(rb, d)) @ (Iinv[b] @ np.cross(rb, d))
                    )
                meff.append(1.0 / k if k > 1e-12 else 0.0)
            row["meff"] = meff
            pen = -gap
            if pen > self.SLOP:
                bias = min(self.ERP * (pen - self.SLOP) / dt, self.MAX_BIAS_VEL)
            elif gap > 0.0:
                # Speculative contact: only stop motion that would penetrate.
                bias = -gap / dt
            else:
                bias = 0.0
            row["bias"] = bias
            row["touching"] = gap < self.TOUCH_BAND
            row["key"] = key
            wp = self._warm.get(key)
            row["imp"] = [wp[0], wp[1], wp[2]] if wp is not None else [0.0, 0.0, 0.0]
            rows.append(row)

        def apply(row, d, mag):
            a, b = row["a"], row["b"]
            imp = mag * d
            v[a] += invm[a] * imp
            om[a] += Iinv[a] @ np.cross(row["ra"], imp)
            if b >= 0:
                v[b] -= invm[b] * imp
                om[b] -= Iinv[b] @ np.cross(row["rb"], imp)

        # Warm start: re-apply the previous step's converged impulses.
        for row in rows:
            for j in range(3):
                if row["imp"][j] != 0.0:
                    apply(row, row["dirs"][j], row["imp"][j])

        for _ in range(self.PGS_ITERS):
            for row in rows:
                a, b = row["a"], row["b"]
                ra, rb = row["ra"], row["rb"]
                dirs = row["dirs"]
                meff = row["meff"]
                imp = row["imp"]
                # --- normal row (Signorini complementarity) ---
                dv = v[a] + np.cross(om[a], ra)
                if b >= 0:
                    dv = dv - v[b] - np.cross(om[b], rb)
                vn = float(dirs[0] @ dv)
                lam = imp[0] - meff[0] * (vn + row["bias"])
                if lam < 0.0:
                    lam = 0.0
                dlam = lam - imp[0]
                if dlam != 0.0:
                    imp[0] = lam
                    apply(row, dirs[0], dlam)
                # --- friction rows (Coulomb cone, box approximation) ---
                if not row["touching"] or imp[0] <= 0.0:
                    if imp[1] != 0.0:
                        apply(row, dirs[1], -imp[1])
                        imp[1] = 0.0
                    if imp[2] != 0.0:
                        apply(row, dirs[2], -imp[2])
                        imp[2] = 0.0
                    continue
                limit = mu * imp[0]
                for j in (1, 2):
                    dv = v[a] + np.cross(om[a], ra)
                    if b >= 0:
                        dv = dv - v[b] - np.cross(om[b], rb)
                    vt = float(dirs[j] @ dv)
                    lam = imp[j] - meff[j] * vt
                    if lam > limit:
                        lam = limit
                    elif lam < -limit:
                        lam = -limit
                    dlam = lam - imp[j]
                    if dlam != 0.0:
                        imp[j] = lam
                        apply(row, dirs[j], dlam)

        for row in rows:
            imp = row["imp"]
            warm_new[row["key"]] = (imp[0], imp[1], imp[2])
        self._warm = warm_new
        for (_a, _b, _p, _n, gap, _k) in contacts:
            if -gap > self.max_penetration:
                self.max_penetration = float(-gap)

        # Numerical safety: bound velocities and scrub non-finite values.
        np.clip(self.v, -50.0, 50.0, out=self.v)
        np.clip(self.omega, -25.0, 25.0, out=self.omega)
        if not np.isfinite(self.v).all():
            self.v[~np.isfinite(self.v)] = 0.0
        if not np.isfinite(self.omega).all():
            self.omega[~np.isfinite(self.omega)] = 0.0

    # ------------------------------------------------------------- reporting
    def _sample_gaps(self) -> np.ndarray:
        """Signed gaps at the frozen contact sample set (fixed order and size)."""
        gaps = np.empty(self.n_gaps, dtype=float)
        idx = 0
        for pair in self.pairs:
            a, b = pair["a"], pair["b"]
            nb = self._R[b] @ pair["normal_b"]
            ref = self._R[b] @ pair["ref_b"] + self.x[b]
            pts_a = pair["corners_a"] @ self._R[a].T + self.x[a]
            for k in range(4):
                kk = min(k, pts_a.shape[0] - 1)
                gaps[idx] = float(nb @ (pts_a[kk] - ref))
                idx += 1
        for (i, k) in self.ground_samples:
            p = self.verts_local[i][k] @ self._R[i].T + self.x[i]
            gaps[idx] = float(p[2])
            idx += 1
        return gaps

    def _frame(self, time: float) -> StateFrame:
        gaps = self._sample_gaps()
        ext_force = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            time=float(max(time, 0.0)),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                # Case-selected constraint: the frozen self-supporting pose of
                # every rigid block (static equilibrium => zero drift target).
                "constraint_value": self.x.copy(),
                "constraint_target": self.x0.copy(),
                "contact_signed_distance": np.ascontiguousarray(gaps),
                "external_force": np.ascontiguousarray(ext_force),
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ public API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1e-12:
            span = target - self.time
            n_sub = max(1, int(round(span / self.SUBSTEP_DT)))
            dt = span / n_sub
            for _ in range(n_sub):
                self._substep(dt)
            self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        drift = float(np.max(np.linalg.norm(self.x - self.x0, axis=1)))
        ke_lin = 0.5 * float(np.sum(self.mass * np.sum(self.v**2, axis=1)))
        ke_ang = 0.0
        for i in range(self.n):
            ke_ang += 0.5 * float(
                self.omega[i] @ (self._R[i] @ self.inertia[i] @ self._R[i].T)
                @ self.omega[i]
            )
        return {
            "method": "semi-implicit Euler + warm-started PGS frictional contacts",
            "substep_dt": self.SUBSTEP_DT,
            "pgs_iterations": self.PGS_ITERS,
            "friction_coefficient": self.mu,
            "substeps_taken": self.steps,
            "sim_time": self.time,
            "max_penetration": self.max_penetration,
            "max_position_drift": drift,
            "kinetic_energy": ke_lin + ke_ang,
        }
