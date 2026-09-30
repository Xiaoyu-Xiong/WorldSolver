"""Self-implemented deformable-solid solver for 026_hard_sphere_ground.

Physical model
--------------
The subject is the frozen closed triangle-mesh shell (icosphere, R = 0.42 m),
discretized with one Lagrangian particle per vertex.  Internal forces:

* membrane edge springs (elastic + Rayleigh damping) built from the rest mesh,
  stiffness ``k_edge = E * h`` with ``h`` the mean rest edge length;
* an enclosed-volume (pressure) force with bulk modulus ``K = E/(3(1-2nu))``
  that resists volumetric compression of the shell;
* a corotational shape-matching force toward the best-fit rigid placement of the
  rest shape (polar decomposition of the mass-weighted covariance), which
  supplies the bulk shear/bending stiffness of a "hard" sphere.  Its damping
  acts only on the deviatoric part of the velocity field, so rigid translation
  and rotation are never artificially damped, and the damping ratio is derived
  from the frozen restitution.

Ground contact (top plane of the frozen ground box) is resolved as a hard
non-penetration constraint after every position update: projection onto the
plane, a normal velocity law using the frozen restitution, and Coulomb friction
on the tangential component.

Time integration is explicit symplectic (semi-implicit) Euler with a substep
limited by the stiffest discrete mode of the assembled model, so states remain
finite, stable and deterministic.  No animation, keyframes, precomputed
trajectories or scene-specific motion tables are used; every sample time is
reached by actually stepping the equations of motion.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from scipy import sparse

from worldsolver.contracts import StateFrame


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edge list of a triangle mesh."""
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0).astype(np.int64)


def _signed_volume(x: np.ndarray, faces: np.ndarray) -> float:
    """Signed enclosed volume of a closed triangle mesh (divergence theorem)."""
    a = x[faces[:, 0]]
    b = x[faces[:, 1]]
    c = x[faces[:, 2]]
    return float(np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6.0)


def _surface_area(x: np.ndarray, faces: np.ndarray) -> float:
    a = x[faces[:, 0]]
    b = x[faces[:, 1]]
    c = x[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


class BenchmarkSolver:
    """Explicit corotational shell solver with hard ground contact."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(scene.config)
        self.config = config

        initial = scene.initial_state()
        self.x0 = np.asarray(initial.position, dtype=float).copy()
        self.x = self.x0.copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.n = int(self.x.shape[0])
        self.total_mass = float(self.mass.sum())
        self.body_id = np.zeros(self.n, dtype=np.int64)

        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.contact_length_scale = float(config.get("contact_length_scale", 0.04))
        self.restitution = float(np.clip(config.get("restitution", 0.5), 0.0, 0.99))
        self.friction = float(max(config.get("friction", 0.0), 0.0))
        self.radius = float(config.get("sphere_radius", 0.42))

        self.faces = self._load_faces()
        self._read_ground()
        self._build_operators()
        self._build_material()
        self._build_gradient_fit()

        self.time = 0.0
        self.max_speed = 60.0
        self.substeps_taken = 0
        self.min_gap = float(np.min(self.x[:, 2] - self.ground_z))
        self.max_abs_velocity = float(np.max(np.abs(self.v)))
        self.contact_started = False

    # ------------------------------------------------------------------ setup
    def _load_faces(self) -> np.ndarray:
        faces = getattr(self.scene, "faces", None)
        if faces is not None:
            return np.asarray(faces, dtype=np.int64)
        asset = str(self.config.get("subject_asset", "assets/subject.npz"))
        candidates = []
        try:
            candidates.append(Path(inspect.getfile(type(self.scene))).resolve().parent / asset)
        except (TypeError, OSError):
            pass
        here = Path(__file__).resolve().parent
        candidates += [here.parent / asset, Path.cwd() / asset, Path(asset)]
        for path in candidates:
            if path.is_file():
                with np.load(path) as data:
                    return np.asarray(data["faces"], dtype=np.int64)
        raise FileNotFoundError(f"Cannot locate subject asset {asset!r}")

    def _read_ground(self) -> None:
        """Top plane and lateral extent of the frozen ground box."""
        self.ground_z = float(self.config.get("ground_z", 0.0))
        self.ground_half = np.array([np.inf, np.inf], dtype=float)
        geometry = self.config.get("scene_geometry") or {}
        for item in geometry.get("objects", []) or []:
            if item.get("semantic") != "boundary" and item.get("id") != "ground":
                continue
            geom = item.get("geometry", {}) or {}
            half = geom.get("half_extents")
            translation = (item.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0])
            if half is not None and len(half) == 3:
                half = np.asarray(half, dtype=float)
                translation = np.asarray(translation, dtype=float)
                self.ground_z = float(translation[2] + half[2])
                self.ground_half = half[:2].copy()
                self.ground_center = translation[:2].copy()
                return
        self.ground_center = np.zeros(2, dtype=float)

    def _build_operators(self) -> None:
        faces = self.faces
        n = self.n
        if _signed_volume(self.x0, faces) < 0.0:
            faces = faces[:, [0, 2, 1]].copy()
            self.faces = faces
        self.rest_volume = _signed_volume(self.x0, faces)
        self.rest_area = _surface_area(self.x0, faces)

        self.edges = _unique_edges(faces)
        ne = int(self.edges.shape[0])
        rows = np.repeat(np.arange(ne, dtype=np.int64), 2)
        cols = self.edges.reshape(-1)
        vals = np.tile(np.array([1.0, -1.0]), ne)
        # D @ x  ->  per-edge difference x[i] - x[j]
        self.D = sparse.csr_matrix((vals, (rows, cols)), shape=(ne, n))
        self.Dt = self.D.transpose().tocsr()
        self.rest_length = np.linalg.norm(self.D @ self.x0, axis=1)
        self.rest_length = np.maximum(self.rest_length, 1e-9)
        self.mean_edge_length = float(self.rest_length.mean())

        # per-face vertex scatter operators (face value -> vertex accumulation)
        nf = int(faces.shape[0])
        face_rows = np.arange(nf, dtype=np.int64)
        ones = np.ones(nf, dtype=float)
        self.S = [
            sparse.csr_matrix((ones, (faces[:, k], face_rows)), shape=(n, nf)) for k in range(3)
        ]

        counts = np.bincount(self.edges.reshape(-1), minlength=n)
        self.max_valence = int(max(counts.max(), 1))

    def _build_material(self) -> None:
        """Map the frozen material constants onto the discrete model."""
        mass_total = self.total_mass
        radius = max(self.radius, 1e-6)
        # Frozen stiffness is a modulus scale (kPa) for this stiff sphere.
        self.young = float(self.config.get("stiffness", 44.0)) * 1.0e3
        poisson = 0.4
        self.bulk = self.young / (3.0 * (1.0 - 2.0 * poisson))

        # Membrane springs: k = E * h with h the mean rest edge length.
        self.k_edge = self.young * self.mean_edge_length
        sum_l2 = float(np.sum(self.rest_length**2))
        # Effective flattening stiffness contributed by the membrane springs for a
        # volume-preserving uniaxial squash of depth delta (s = delta / 2R).
        k_flat_edge = 0.05 * self.k_edge * sum_l2 / (radius**2)
        # Target overall contact stiffness of a stiff elastic sphere: K ~ E * R.
        k_flat_target = self.young * radius
        k_flat_shape = max(k_flat_target - k_flat_edge, 0.25 * k_flat_target)
        # Shape-matching energy of the same squash gives K = ks * M / 8.
        self.k_shape = 8.0 * k_flat_shape / max(mass_total, 1e-9)
        self.k_flat_total = k_flat_edge + k_flat_shape

        # Damping ratio implied by the frozen restitution (log-decrement law).
        log_e = float(np.log(max(self.restitution, 1e-3)))
        zeta = -log_e / float(np.sqrt(np.pi**2 + log_e**2))
        self.zeta = float(np.clip(zeta, 0.03, 0.5))
        self.c_shape = 2.0 * self.zeta * float(np.sqrt(self.k_shape))
        self.c_global = float(max(self.config.get("damping", 0.0), 0.0))

        mass_mean = float(self.mass.mean())
        self.c_edge = 2.0 * self.zeta * float(np.sqrt(self.k_edge * mass_mean))

        # Pressure (breathing) mode in the volume coordinate.
        self.k_vol = self.bulk / max(self.rest_volume, 1e-9)
        gen_mass_vol = mass_total / max(self.rest_area**2, 1e-12)
        self.c_vol = 2.0 * 0.3 * float(np.sqrt(self.k_vol * gen_mass_vol))

        # Stability: substep from the stiffest assembled mode and damping rate.
        mass_min = float(max(self.mass.min(), 1e-12))
        omega_edge = np.sqrt(2.0 * self.max_valence * self.k_edge / mass_min)
        omega_shape = np.sqrt(self.k_shape)
        omega_vol = np.sqrt(self.k_vol / gen_mass_vol) if gen_mass_vol > 0.0 else 0.0
        omega_max = float(max(omega_edge, omega_shape, omega_vol, 1.0))
        rate_edge = 2.0 * self.max_valence * self.c_edge / mass_min
        rate_shape = self.c_shape + self.c_global
        rate_vol = self.c_vol * self.rest_area**2 / max(mass_total, 1e-9)
        rate_max = float(max(rate_edge, rate_shape, rate_vol, 1.0))
        self.dt_max = float(min(0.2 / omega_max, 1.0 / rate_max, 1.0e-4))
        self.omega_max = omega_max

    def _build_gradient_fit(self) -> None:
        """Least-squares per-vertex deformation-gradient operator (rest space)."""
        pairs = np.vstack([self.edges, self.edges[:, ::-1]]).astype(np.int64)
        self.fit_i = pairs[:, 0].copy()
        self.fit_j = pairs[:, 1].copy()
        rest_edge = self.x0[self.fit_j] - self.x0[self.fit_i]
        self.fit_w = 1.0 / np.maximum(np.einsum("ij,ij->i", rest_edge, rest_edge), 1e-18)
        self.fit_rest_edge = rest_edge

        center0 = np.average(self.x0, axis=0, weights=self.mass)
        self.fit_rest_center = center0[None, :] - self.x0
        self.fit_wc = 1.0 / np.maximum(
            np.einsum("ij,ij->i", self.fit_rest_center, self.fit_rest_center), 1e-18
        )

        moment = np.zeros((self.n, 3, 3), dtype=float)
        np.add.at(
            moment,
            self.fit_i,
            self.fit_w[:, None, None] * rest_edge[:, :, None] * rest_edge[:, None, :],
        )
        moment += (
            self.fit_wc[:, None, None]
            * self.fit_rest_center[:, :, None]
            * self.fit_rest_center[:, None, :]
        )
        trace = np.trace(moment, axis1=1, axis2=2)
        ridge = np.maximum(1e-10 * trace, 1e-14)
        moment += ridge[:, None, None] * np.eye(3)[None, :, :]
        self.fit_inverse = np.linalg.inv(moment)

        # shape-matching rest configuration (mass-centered)
        self.rest_com = center0
        self.q = self.x0 - center0[None, :]
        self.rest_inertia_scale = float(np.sum(self.mass * np.einsum("ij,ij->i", self.q, self.q)))
        self.rest_threshold = 0.3

    # ----------------------------------------------------------------- physics
    def _forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Assemble gravity + membrane + pressure + corotational shape forces."""
        f = self.mass[:, None] * self.gravity[None, :]

        # --- membrane edge springs with Rayleigh damping -----------------
        de = self.D @ x
        length = np.linalg.norm(de, axis=1)
        length_safe = np.maximum(length, 1e-12)
        direction = de / length_safe[:, None]
        rate = np.einsum("ij,ij->i", self.D @ v, direction)
        tension = self.k_edge * (length - self.rest_length) + self.c_edge * rate
        f -= self.Dt @ (tension[:, None] * direction)

        # --- enclosed-volume (pressure) force ---------------------------
        a = x[self.faces[:, 0]]
        b = x[self.faces[:, 1]]
        c = x[self.faces[:, 2]]
        volume = float(np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6.0)
        grad = (
            self.S[0] @ (np.cross(b, c) / 6.0)
            + self.S[1] @ (np.cross(c, a) / 6.0)
            + self.S[2] @ (np.cross(a, b) / 6.0)
        )
        volume_rate = float(np.einsum("ij,ij->", grad, v))
        pressure = self.k_vol * (self.rest_volume - volume) - self.c_vol * volume_rate
        f += pressure * grad

        # --- corotational shape matching --------------------------------
        com = (self.mass[:, None] * x).sum(axis=0) / self.total_mass
        p = x - com[None, :]
        covariance = (self.mass[:, None] * p).T @ self.q
        try:
            u, _s, vt = np.linalg.svd(covariance)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u = u.copy()
                u[:, -1] *= -1.0
                rotation = u @ vt
        except np.linalg.LinAlgError:
            rotation = np.eye(3)
        goal = com[None, :] + self.q @ rotation.T
        f += (self.k_shape * self.mass)[:, None] * (goal - x)

        # --- deviatoric (non-rigid) velocity damping --------------------
        v_com = (self.mass[:, None] * v).sum(axis=0) / self.total_mass
        v_rel = v - v_com[None, :]
        angular_momentum = (self.mass[:, None] * np.cross(p, v_rel)).sum(axis=0)
        inertia = np.eye(3) * float(
            np.sum(self.mass * np.einsum("ij,ij->i", p, p))
        ) - np.einsum("i,ij,ik->jk", self.mass, p, p)
        inertia = inertia + np.eye(3) * (1e-8 * max(self.rest_inertia_scale, 1e-12))
        try:
            omega = np.linalg.solve(inertia, angular_momentum)
        except np.linalg.LinAlgError:
            omega = np.zeros(3)
        v_rigid = v_com[None, :] + np.cross(np.broadcast_to(omega, p.shape), p)
        f -= ((self.c_shape + self.c_global) * self.mass)[:, None] * (v - v_rigid)

        self.last_volume = volume
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def _resolve_contact(self, x: np.ndarray, v: np.ndarray) -> None:
        """Hard non-penetration constraint on the ground plane + friction."""
        gap = x[:, 2] - self.ground_z
        if np.isfinite(self.ground_half).all():
            inside = (np.abs(x[:, 0] - self.ground_center[0]) <= self.ground_half[0]) & (
                np.abs(x[:, 1] - self.ground_center[1]) <= self.ground_half[1]
            )
        else:
            inside = np.ones(self.n, dtype=bool)
        hit = inside & (gap < 0.0)
        if not np.any(hit):
            return
        self.contact_started = True
        x[hit, 2] = self.ground_z
        index = np.flatnonzero(hit & (v[:, 2] < 0.0))
        if index.size == 0:
            return
        vn = v[index, 2]
        bounce = np.where(-vn > self.rest_threshold, -self.restitution * vn, 0.0)
        normal_impulse = self.mass[index] * (bounce - vn)
        v[index, 2] = bounce
        tangent = v[index, :2]
        speed = np.linalg.norm(tangent, axis=1)
        active = speed > 1e-9
        if np.any(active):
            reduction = np.zeros_like(speed)
            reduction[active] = (self.friction * normal_impulse[active]) / (
                self.mass[index][active] * speed[active]
            )
            scale = np.clip(1.0 - reduction, 0.0, 1.0)
            v[index, :2] = tangent * scale[:, None]

    def _step(self, dt: float) -> None:
        force = self._forces(self.x, self.v)
        self.v += dt * force / self.mass[:, None]
        np.clip(self.v, -self.max_speed, self.max_speed, out=self.v)
        self.x += dt * self.v
        self._resolve_contact(self.x, self.v)
        self.substeps_taken += 1

    # ------------------------------------------------------------------ output
    def _deformation_gradient(self) -> np.ndarray:
        current_edge = self.x[self.fit_j] - self.x[self.fit_i]
        com = (self.mass[:, None] * self.x).sum(axis=0) / self.total_mass
        current_center = com[None, :] - self.x
        rhs = np.zeros((self.n, 3, 3), dtype=float)
        np.add.at(
            rhs,
            self.fit_i,
            self.fit_w[:, None, None]
            * current_edge[:, :, None]
            * self.fit_rest_edge[:, None, :],
        )
        rhs += (
            self.fit_wc[:, None, None]
            * current_center[:, :, None]
            * self.fit_rest_center[:, None, :]
        )
        gradient = rhs @ self.fit_inverse
        return np.nan_to_num(gradient, nan=0.0, posinf=1.0e3, neginf=-1.0e3)

    def _frame(self, time: float) -> StateFrame:
        gap = self.x[:, 2] - self.ground_z
        gradient = self._deformation_gradient()
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": gradient,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": gap.copy(),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        self.min_gap = min(self.min_gap, float(gap.min()))
        self.max_abs_velocity = max(self.max_abs_velocity, float(np.abs(self.v).max()))
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        remaining = target - self.time
        if remaining > 1.0e-15:
            steps = int(np.ceil(remaining / self.dt_max))
            dt = remaining / float(max(steps, 1))
            for _ in range(max(steps, 1)):
                self._step(dt)
        self.time = max(self.time, target)
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit corotational shell (edge springs + pressure + shape matching)",
            "substeps": int(self.substeps_taken),
            "substep_dt": float(self.dt_max),
            "young_modulus_pa": float(self.young),
            "k_edge": float(self.k_edge),
            "k_shape": float(self.k_shape),
            "k_volume": float(self.k_vol),
            "damping_ratio": float(self.zeta),
            "effective_contact_stiffness": float(self.k_flat_total),
            "min_contact_gap": float(self.min_gap),
            "max_abs_velocity": float(self.max_abs_velocity),
            "contact_detected": bool(self.contact_started),
            "final_volume_ratio": float(
                getattr(self, "last_volume", self.rest_volume) / max(self.rest_volume, 1e-12)
            ),
        }

