"""Mass-spring plastic shell solver for the ductile hollow ball drop.

The solver integrates a triangular-mesh particle shell with gravity, floor
contact, internal spring forces, plastic rest-length flow, and strain-based
fracture.  It produces per-sample deformation gradients, external forces, mass,
velocity and signed floor distance at the requested sample times.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _parse_obj_faces(obj_path: Path) -> list[list[int]]:
    """Read face indices (1-based) from a Wavefront OBJ file."""
    faces: list[list[int]] = []
    with obj_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("f "):
                continue
            parts = line.split()[1:]
            # take the vertex index of each "v/vt/vn" token
            verts = [int(p.split("/")[0]) for p in parts]
            if len(verts) < 3:
                continue
            # triangulate a fan
            for k in range(1, len(verts) - 1):
                faces.append([verts[0], verts[k], verts[k + 1]])
    return faces


def _unique_edges(faces: list[list[int]]) -> np.ndarray:
    """Return an (E,2) array of unique undirected edges with the smaller index first."""
    edge_set: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            edge_set.add((min(u, v), max(u, v)))
    return np.array(sorted(edge_set), dtype=np.int64)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Time-stepping parameters.  sample_dt is exactly 1/30 s; choose a
        # stable explicit substep and snap it to an integer division.
        self.sample_dt = float(self.config["sample_dt"])
        self.target_dt = 5.0e-5
        self.n_sub = max(1, int(round(self.sample_dt / self.target_dt)))
        self.dt = self.sample_dt / self.n_sub

        # Initial particle data (the frozen scene gives us a hollow shell).
        frame0 = scene.initial_state()
        self.x = np.asarray(frame0.position, dtype=float).copy()
        self.v = np.asarray(frame0.velocity, dtype=float).copy()
        self.m = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = (
            np.asarray(frame0.body_id, dtype=int).copy()
            if frame0.body_id is not None
            else np.arange(len(self.x), dtype=int)
        )
        self.rest_pos = self.x.copy()
        self.time = 0.0

        # Material / contact parameters (SI units, tuned for a stable 7 s run).
        self.k_shell = 1.0e6          # shell spring stiffness [N/m]
        self.zeta = 0.25              # spring damping ratio
        self.k_ground = 5.0e6         # floor penalty stiffness [N/m]
        self.zeta_ground = 0.6       # floor contact damping ratio
        self.mu_friction = 0.35       # Coulomb-ish friction coefficient
        self.v_max = 50.0             # velocity safety clamp [m/s]

        # Plasticity / fracture.
        self.yield_strain = 0.04      # elastic strain before plastic flow
        self.plastic_rate = 8.0       # speed of rest-length plastic flow [1/s]
        self.fracture_strain = 0.30   # tensile strain at which an edge breaks
        self.min_lp_factor = 0.15     # prevent zero-length springs

        # Build edges from the mesh asset.
        obj_path = Path(__file__).resolve().parent.parent / "assets" / "hollow_ball.obj"
        if not obj_path.is_file():
            # Fallback if the runner ever changes the working directory.
            obj_path = Path("assets/hollow_ball.obj").resolve()
        faces = _parse_obj_faces(obj_path)
        self.edges = _unique_edges(faces)

        i = self.edges[:, 0]
        j = self.edges[:, 1]
        dx0 = self.rest_pos[j] - self.rest_pos[i]
        l0 = np.linalg.norm(dx0, axis=1)
        valid = l0 > 1e-6
        self.edges = self.edges[valid]
        self.L0 = l0[valid]
        self.Lp = self.L0.copy()
        self.broken = np.zeros(len(self.Lp), dtype=bool)

        # Neighbor graph for per-particle deformation gradients.
        self.neighbors: list[np.ndarray] = [np.empty(0, dtype=int) for _ in range(len(self.x))]
        for a, b in self.edges:
            # a,b are the edge endpoints; append to neighbor list later
            self.neighbors[a] = np.append(self.neighbors[a], b)
            self.neighbors[b] = np.append(self.neighbors[b], a)
        for i in range(len(self.x)):
            self.neighbors[i] = np.unique(self.neighbors[i])

        # Derived damping coefficients.
        m_avg = float(np.mean(self.m))
        self.c_spring = 2.0 * self.zeta * np.sqrt(m_avg * self.k_shell)
        self.c_ground = 2.0 * self.zeta_ground * np.sqrt(m_avg * self.k_ground)

        # Cache the scalar / attribute skeleton from the initial frame.
        self._base_scalars = dict(frame0.scalars)
        self._base_attrs = {
            name: np.asarray(value).copy()
            for name, value in frame0.attributes.items()
        }

        self._last_ext = np.zeros_like(self.x)

    def _floor_contact_force(self, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return contact force and signed distance to the ground plane z=0."""
        signed = x[:, 2].copy()
        contact = np.zeros_like(x)

        penetrate = signed < 0.0
        if not np.any(penetrate):
            return contact, signed

        depth = -signed[penetrate]
        v_p = v[penetrate]

        # Normal penalty + damping (only when moving toward the floor).
        v_z = np.minimum(v_p[:, 2], 0.0)
        f_n = self.k_ground * depth - self.c_ground * v_z
        f_n = np.maximum(f_n, 0.0)

        contact[penetrate, 2] = f_n

        # Tangential Coulomb friction (proportional to normal force, limited).
        v_t = v_p.copy()
        v_t[:, 2] = 0.0
        v_t_mag = np.linalg.norm(v_t, axis=1)
        f_t_mag = self.mu_friction * f_n
        scale = np.where(v_t_mag > 1e-8, f_t_mag / v_t_mag, 0.0)
        scale = np.minimum(scale, 1.0)  # safety
        contact[penetrate, :2] = -scale[:, None] * v_t[:, :2]

        return contact, signed

    def _internal_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute spring/damper forces for unbroken edges; update plastic lengths."""
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        dx = self.x[j] - self.x[i]
        l = np.linalg.norm(dx, axis=1)
        l = np.maximum(l, 1e-8)
        e = dx / l[:, None]

        active = ~self.broken

        # Plastic flow: shift rest length toward current length once the
        # elastic strain exceeds the yield threshold.
        elast = (l - self.Lp) / self.L0
        over = np.abs(elast) > self.yield_strain
        dlp = np.zeros_like(self.Lp)
        mask = active & over
        dlp[mask] = (
            self.plastic_rate
            * (np.sign(elast[mask]) * (np.abs(elast[mask]) - self.yield_strain))
            * self.L0[mask]
            * self.dt
        )
        self.Lp += dlp
        self.Lp = np.clip(self.Lp, self.min_lp_factor * self.L0, 2.0 * self.L0)

        # Tensile fracture after sufficient deformation.
        total_strain = (l - self.L0) / self.L0
        break_mask = (total_strain > self.fracture_strain) & active
        self.broken |= break_mask
        active &= ~break_mask

        # Spring force (zero on broken edges).
        f_mag = np.where(active, self.k_shell * (l - self.Lp), 0.0)
        f = f_mag[:, None] * e

        # Edge damping proportional to relative velocity along the edge.
        dv = self.v[j] - self.v[i]
        fdamp = -self.c_spring * np.sum(dv * e, axis=1)[:, None] * e
        f += fdamp

        # Scatter forces onto particles.
        n = len(self.x)
        force = np.zeros_like(self.x)
        for dim in range(3):
            force[:, dim] = np.bincount(
                i, weights=-f[:, dim], minlength=n
            ) + np.bincount(j, weights=f[:, dim], minlength=n)
        return force, active

    def _substep(self) -> None:
        # External forces: gravity + floor contact.
        ext = (self.m[:, None] * self.gravity).astype(float)
        contact, _ = self._floor_contact_force(self.x, self.v)
        ext += contact
        self._last_ext = ext.copy()

        # Internal spring forces (also updates plastic state).
        f_int, _ = self._internal_forces()
        ext += f_int

        # Explicit symplectic Euler step.
        a = ext / self.m[:, None]
        self.v += a * self.dt

        # Global velocity damping for numerical stability.
        self.v *= 1.0 - 0.1 * self.dt

        # Safety clamp to avoid runaway velocities.
        speed = np.linalg.norm(self.v, axis=1)
        over = speed > self.v_max
        if np.any(over):
            self.v[over] = (self.v[over].T / speed[over]).T * self.v_max

        self.x += self.v * self.dt

    def _deformation_gradient(self) -> np.ndarray:
        """Compute per-particle deformation gradient via local least squares."""
        n = len(self.x)
        F = np.tile(np.eye(3, dtype=float), (n, 1, 1))
        for i in range(n):
            nbr = self.neighbors[i]
            if len(nbr) < 3:
                continue
            X = self.rest_pos[nbr] - self.rest_pos[i]
            x = self.x[nbr] - self.x[i]
            # inverse-distance weights give a smoother estimate across irregular neighborhoods
            w = 1.0 / (np.linalg.norm(X, axis=1) + 1e-6) ** 2
            # Dm = sum_k w_k (X_k)(X_k)^T  [3x3]
            Dm = np.einsum("k,ki,kj->ij", w, X, X)
            # Ds = sum_k w_k (x_k)(X_k)^T  [3x3]
            Ds = np.einsum("k,ki,kj->ij", w, x, X)
            try:
                Dm_inv = np.linalg.inv(Dm + 1e-8 * np.eye(3))
            except np.linalg.LinAlgError:
                continue
            F[i] = Ds @ Dm_inv
        return F

    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            raise ValueError("advance_to must be called with non-decreasing times")

        # Step forward to the requested time.
        while self.time < time - 0.5 * self.dt:
            self._substep()
            self.time += self.dt

        # Build the output StateFrame at exactly the requested time.
        F = self._deformation_gradient()
        contact_force, signed_dist = self._floor_contact_force(self.x, self.v)
        ext_force = self.m[:, None] * self.gravity + contact_force

        attributes = {
            "mass_quantity": self.m.copy(),
            "deformation_gradient": F,
        }
        # carry forward any non-conflicting base attributes (particle_radius, etc.)
        for name, value in self._base_attrs.items():
            if name not in attributes:
                attributes[name] = value.copy()

        fields = {
            "contact_signed_distance": signed_dist,
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        # include any base fields the scene provided
        if hasattr(self, "_base_fields"):
            for name, value in self._base_fields.items():
                if name not in fields:
                    fields[name] = value.copy()

        scalars = dict(self._base_scalars)
        scalars.setdefault("contact_length_scale", float(self.config["sampling"]["spacing"]))

        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps_per_sample": self.n_sub,
            "dt": self.dt,
            "n_particles": len(self.x),
            "n_edges": len(self.edges),
            "broken_edges": int(np.sum(self.broken)),
        }
