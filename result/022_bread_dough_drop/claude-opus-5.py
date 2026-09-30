"""Viscoplastic FEM solver for the bread-dough drop scene.

Physical model
--------------
The frozen subject asset is a closed triangle surface mesh.  A tetrahedral
volume mesh is generated deterministically by connecting every surface
triangle to the interior centroid node (star decomposition).  The dough is
then advanced with an explicit (symplectic Euler) corotational finite element
method using a multiplicative elastoplastic decomposition

    F = Fe @ Fp ,

where the elastic part drives a corotational stress and the plastic part
absorbs deviatoric strain beyond a von Mises yield surface at a finite flow
rate (the model used for large viscoplastic flow of dough-like materials).
Contact with the frozen ground box is resolved with a velocity-level
non-penetration constraint plus Coulomb friction, so the body impacts,
flattens plastically, recovers elastically a little, and keeps a permanent
footprint.

Only NumPy/SciPy are used; no external simulator is involved.
"""
from __future__ import annotations

import numpy as np

try:  # SciPy is used only for a fast scatter (sum) operator.
    from scipy.sparse import csr_matrix

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - fall back to np.add.at
    _HAVE_SCIPY = False


class BenchmarkSolver:
    """Explicit corotational-FEM viscoplastic solid solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config

        frame0 = scene.initial_state()
        self.surface_x0 = np.asarray(frame0.position, dtype=float).copy()
        self.n_surface = int(self.surface_x0.shape[0])

        self.export_mass = np.asarray(frame0.mass, dtype=float).copy()
        if self.export_mass.shape != (self.n_surface,):
            total = float(config.get("subject_mass", 1.0))
            self.export_mass = np.full(self.n_surface, total / self.n_surface)
        self.total_mass = float(np.sum(self.export_mass))

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.ground_z = float(config.get("ground_z", 0.0))
        self.contact_length_scale = float(config.get("contact_length_scale", 0.02))
        self.restitution = float(np.clip(config.get("restitution", 0.0), 0.0, 0.95))
        self.friction = float(max(config.get("friction", 0.5), 0.0))
        self.damping_cfg = float(max(config.get("damping", 1.0), 0.0))
        self.stiffness_cfg = float(max(config.get("stiffness", 1.0), 1e-3))
        self.plasticity_cfg = float(np.clip(config.get("plasticity", 0.5), 0.0, 1.0))

        # ---- topology: surface faces -> star tetrahedralization
        faces = self._load_faces()
        verts = self.surface_x0
        # Ensure outward winding (positive enclosed volume).
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        enclosed = np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0
        if enclosed < 0.0:
            faces = faces[:, [0, 2, 1]]

        centroid = verts.mean(axis=0)
        self.x0 = np.vstack([verts, centroid[None, :]])
        self.n_nodes = int(self.x0.shape[0])
        center_index = self.n_nodes - 1

        tets = np.concatenate(
            [faces, np.full((faces.shape[0], 1), center_index, dtype=np.int64)], axis=1
        )
        Dm = self._edge_matrix(self.x0, tets)
        det = np.linalg.det(Dm)
        flip = det < 0.0
        if np.any(flip):
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            Dm = self._edge_matrix(self.x0, tets)
            det = np.linalg.det(Dm)
        # Drop degenerate (zero-volume) elements.
        keep = np.abs(det) > 1e-12
        tets = tets[keep]
        Dm = Dm[keep]
        det = det[keep]
        self.tets = np.ascontiguousarray(tets)
        self.n_tets = int(self.tets.shape[0])
        self.rest_volume = det / 6.0
        self.Bm = np.linalg.inv(Dm)
        self.total_volume = float(np.sum(self.rest_volume))
        self.density = self.total_mass / max(self.total_volume, 1e-9)

        # ---- lumped nodal masses (physically consistent volume weighting)
        node_mass = np.zeros(self.n_nodes)
        share = np.repeat((self.density * self.rest_volume / 4.0)[:, None], 4, axis=1)
        np.add.at(node_mass, self.tets.ravel(), share.ravel())
        floor = 1e-6 * self.total_mass / self.n_nodes
        self.node_mass = np.maximum(node_mass, max(floor, 1e-12))
        self.node_mass *= self.total_mass / float(np.sum(self.node_mass))
        self.inv_node_mass = 1.0 / self.node_mass

        self.incidence = np.bincount(self.tets.ravel(), minlength=self.n_nodes).astype(float)
        self.incidence = np.maximum(self.incidence, 1.0)

        if _HAVE_SCIPY:
            rows = self.tets.ravel()
            cols = np.arange(rows.size)
            self.scatter = csr_matrix(
                (np.ones(rows.size), (rows, cols)), shape=(self.n_nodes, rows.size)
            )
        else:  # pragma: no cover
            self.scatter = None

        # ---- time step (deterministic, CFL-limited)
        sample_times = [float(t) for t in scene.sample_times()]
        self.sample_times = sample_times
        if len(sample_times) > 1:
            self.sample_dt = float(sample_times[1] - sample_times[0])
        else:
            self.sample_dt = float(config.get("sample_dt", 1.0 / 30.0))
        self.substeps = 340
        self.dt = self.sample_dt / self.substeps

        # ---- material parameters (bounded by the explicit stability limit)
        face_a = 0.5 * np.linalg.norm(
            np.cross(
                self.x0[self.tets[:, 1]] - self.x0[self.tets[:, 0]],
                self.x0[self.tets[:, 2]] - self.x0[self.tets[:, 0]],
            ),
            axis=1,
        )
        face_b = 0.5 * np.linalg.norm(
            np.cross(
                self.x0[self.tets[:, 1]] - self.x0[self.tets[:, 0]],
                self.x0[self.tets[:, 3]] - self.x0[self.tets[:, 0]],
            ),
            axis=1,
        )
        face_c = 0.5 * np.linalg.norm(
            np.cross(
                self.x0[self.tets[:, 2]] - self.x0[self.tets[:, 0]],
                self.x0[self.tets[:, 3]] - self.x0[self.tets[:, 0]],
            ),
            axis=1,
        )
        face_d = 0.5 * np.linalg.norm(
            np.cross(
                self.x0[self.tets[:, 2]] - self.x0[self.tets[:, 1]],
                self.x0[self.tets[:, 3]] - self.x0[self.tets[:, 1]],
            ),
            axis=1,
        )
        max_face = np.maximum(np.maximum(face_a, face_b), np.maximum(face_c, face_d))
        altitude = 3.0 * np.abs(self.rest_volume) / np.maximum(max_face, 1e-12)
        h_min = float(max(np.percentile(altitude, 2.0), 1e-3))
        self.h_min = h_min

        poisson = 0.33
        e_request = 1000.0 * self.stiffness_cfg  # config stiffness in kPa-like units
        c_limit = 0.25 * h_min / self.dt
        e_cfl = self.density * c_limit * c_limit / 1.482  # (lambda + 2 mu) = 1.482 E
        young = float(min(e_request, e_cfl))
        young = float(max(young, 1e-3))
        self.young = young
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.wave_speed = float(np.sqrt((self.lam + 2.0 * self.mu) / max(self.density, 1e-12)))

        # Yield stress: higher configured plasticity -> easier plastic flow.
        self.yield_stress = 260.0 * (1.0 - 0.5 * self.plasticity_cfg)
        self.yield_strain = float(
            np.clip(self.yield_stress / max(2.0 * self.mu, 1e-9), 0.005, 0.35)
        )
        self.flow_rate = 200.0 * max(self.plasticity_cfg, 0.05)  # 1/s
        self.plastic_alpha = float(np.clip(self.flow_rate * self.dt, 0.0, 1.0))
        self.plastic_cap = 2.5  # work-hardening limit on accumulated plastic strain

        # Rigid-motion-invariant Kelvin-Voigt-like relative velocity damping.
        self.damp_rate = float(np.clip(15.0 * self.damping_cfg, 0.0, 0.5 / self.dt))

        # ---- state
        self.x = self.x0.copy()
        v0_cfg = np.asarray(config.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0_cfg.reshape(1, 3), (self.n_nodes, 1))
        self.Fp_inv = np.tile(np.eye(3), (self.n_tets, 1, 1))
        self.plastic_strain = np.zeros(self.n_tets)
        self.step_index = 0
        self.time = 0.0
        self.max_speed = 60.0
        self._diag = {
            "method": "explicit corotational FEM with von Mises plastic flow",
            "nodes": self.n_nodes,
            "elements": self.n_tets,
            "dt": self.dt,
            "substeps_per_sample": self.substeps,
            "density": self.density,
            "young_modulus_pa": self.young,
            "shear_modulus_pa": self.mu,
            "lame_lambda_pa": self.lam,
            "yield_stress_pa": self.yield_stress,
            "yield_strain": self.yield_strain,
            "flow_rate_per_s": self.flow_rate,
            "damping_rate_per_s": self.damp_rate,
            "wave_speed_m_per_s": self.wave_speed,
            "rest_volume_m3": self.total_volume,
            "min_altitude_m": self.h_min,
            "seed": self.seed,
        }

    # -------------------------------------------------------------- utilities
    def _load_faces(self) -> np.ndarray:
        from pathlib import Path

        asset = self.config.get("subject_asset", "assets/subject.npz")
        candidates = []
        path = Path(str(asset))
        if path.is_absolute():
            candidates.append(path)
        scene_dir = None
        try:
            import inspect

            scene_dir = Path(inspect.getfile(type(self.scene))).resolve().parent
        except Exception:
            scene_dir = None
        if scene_dir is not None:
            candidates.append(scene_dir / str(asset))
        here = Path(__file__).resolve().parent
        candidates.append(here.parent / str(asset))
        candidates.append(Path.cwd() / str(asset))
        for candidate in candidates:
            try:
                if candidate.is_file():
                    with np.load(candidate) as data:
                        for key in ("faces", "triangles", "indices", "elements"):
                            if key in data.files:
                                faces = np.asarray(data[key], dtype=np.int64)
                                if faces.ndim == 2 and faces.shape[1] == 3:
                                    return np.ascontiguousarray(faces)
            except Exception:
                continue
        raise RuntimeError("Subject triangle topology could not be loaded")

    @staticmethod
    def _edge_matrix(x: np.ndarray, tets: np.ndarray) -> np.ndarray:
        e1 = x[tets[:, 1]] - x[tets[:, 0]]
        e2 = x[tets[:, 2]] - x[tets[:, 0]]
        e3 = x[tets[:, 3]] - x[tets[:, 0]]
        return np.stack([e1, e2, e3], axis=2)

    def _sum_to_nodes(self, contrib: np.ndarray) -> np.ndarray:
        """Sum per-element per-corner values (T,4,C) onto nodes (n_nodes,C)."""
        flat = np.ascontiguousarray(contrib).reshape(self.n_tets * 4, -1)
        if self.scatter is not None:
            return np.asarray(self.scatter @ flat)
        out = np.zeros((self.n_nodes, flat.shape[1]))
        np.add.at(out, self.tets.ravel(), flat)
        return out

    # ------------------------------------------------------------- one step
    def _deformation(self):
        Ds = self._edge_matrix(self.x, self.tets)
        F = Ds @ self.Bm
        Fe = F @ self.Fp_inv
        U, S, Vt = np.linalg.svd(Fe)
        # Fix reflections so the rotation part is proper.
        det_uv = np.linalg.det(U) * np.linalg.det(Vt)
        flip = det_uv < 0.0
        if np.any(flip):
            U[flip, :, 2] = -U[flip, :, 2]
            S[flip, 2] = -S[flip, 2]
        S_clamped = np.clip(S, 0.08, 12.0)
        return F, Fe, U, S_clamped, Vt

    def _step(self) -> None:
        dt = self.dt
        F, Fe, U, S, Vt = self._deformation()

        R = U @ Vt
        Fe_s = np.einsum("nij,nj,njk->nik", U, S, Vt)

        # Corotational first Piola-Kirchhoff stress of the elastic part.
        trace = np.einsum("nji,nji->n", R, Fe_s) - 3.0
        P = 2.0 * self.mu * (Fe_s - R) + (self.lam * trace)[:, None, None] * R

        G = self.Bm @ self.Fp_inv
        H = -(self.rest_volume[:, None, None]) * (P @ np.transpose(G, (0, 2, 1)))

        contrib = np.empty((self.n_tets, 4, 3))
        f1 = H[:, :, 0]
        f2 = H[:, :, 1]
        f3 = H[:, :, 2]
        contrib[:, 1, :] = f1
        contrib[:, 2, :] = f2
        contrib[:, 3, :] = f3
        contrib[:, 0, :] = -(f1 + f2 + f3)

        # Relative-velocity (viscous) damping: invariant under rigid translation.
        if self.damp_rate > 0.0:
            v_tet = self.v[self.tets]
            v_mean = v_tet.mean(axis=1)
            dv = v_tet - v_mean[:, None, :]
            weight = (self.node_mass[self.tets] / self.incidence[self.tets])[:, :, None]
            contrib -= self.damp_rate * weight * dv

        force = self._sum_to_nodes(contrib)
        force += self.node_mass[:, None] * self.gravity[None, :]
        force = np.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)

        # Symplectic Euler velocity update.
        self.v += dt * force * self.inv_node_mass[:, None]
        np.clip(self.v, -self.max_speed, self.max_speed, out=self.v)

        self._resolve_ground_contact(dt)

        self.x += dt * self.v
        # Hard non-penetration safety net against the frozen ground plane.
        below = self.x[:, 2] < self.ground_z
        if np.any(below):
            self.x[below, 2] = self.ground_z

        self._plastic_update(F, U, S, Vt)
        self.step_index += 1
        self.time = self.step_index * dt

    def _resolve_ground_contact(self, dt: float) -> None:
        z = self.x[:, 2]
        vz = self.v[:, 2]
        gap = z - self.ground_z
        # Predictive contact set: samples that would penetrate this step.
        active = (gap + dt * vz) < 0.0
        if not np.any(active):
            return
        idx = np.nonzero(active)[0]
        vz_a = vz[idx]
        # Velocity that lands exactly on the surface (never pulls samples down).
        vz_allowed = -gap[idx] / dt
        vz_new = np.maximum(-self.restitution * vz_a, vz_allowed)
        vz_new = np.maximum(vz_new, vz_a)  # only push, never suck downward
        dvn = vz_new - vz_a
        self.v[idx, 2] = vz_new
        if self.friction > 0.0:
            vt = self.v[idx, :2]
            speed = np.linalg.norm(vt, axis=1)
            scale = np.clip(1.0 - self.friction * dvn / np.maximum(speed, 1e-12), 0.0, 1.0)
            self.v[idx, :2] = vt * scale[:, None]

    def _plastic_update(self, F: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> None:
        if self.plastic_alpha <= 0.0:
            return
        eps = np.log(S)
        mean = eps.mean(axis=1, keepdims=True)
        dev = eps - mean
        norm = np.linalg.norm(dev, axis=1)
        excess = norm - self.yield_strain
        flowing = (excess > 0.0) & (self.plastic_strain < self.plastic_cap)
        if not np.any(flowing):
            return
        gamma = np.zeros_like(norm)
        gamma[flowing] = self.plastic_alpha * excess[flowing]
        gamma = np.minimum(gamma, np.maximum(norm - 1e-9, 0.0))
        eps_new = eps - (gamma / np.maximum(norm, 1e-12))[:, None] * dev
        ratio = np.exp(eps_new - eps)
        # Fe_new = Fe @ (V diag(ratio) V^T)  =>  Fp_inv_new = Fp_inv @ (same)
        delta = np.einsum("nji,nj,njk->nik", Vt, ratio, Vt)
        self.Fp_inv = self.Fp_inv @ delta
        # Plastic flow is deviatoric: renormalize any determinant drift.
        det = np.linalg.det(self.Fp_inv)
        sign = np.where(det < 0.0, -1.0, 1.0)
        scale = sign * np.cbrt(np.maximum(np.abs(det), 1e-12))
        self.Fp_inv /= scale[:, None, None]
        self.Fp_inv = np.nan_to_num(self.Fp_inv, nan=0.0, posinf=0.0, neginf=0.0)
        self.plastic_strain += gamma

    # ---------------------------------------------------------------- output
    def _node_deformation_gradient(self) -> np.ndarray:
        Ds = self._edge_matrix(self.x, self.tets)
        F = Ds @ self.Bm
        contrib = np.repeat(F.reshape(self.n_tets, 1, 9), 4, axis=1)
        summed = self._sum_to_nodes(contrib)
        per_node = summed / self.incidence[:, None]
        out = per_node[: self.n_surface].reshape(self.n_surface, 3, 3)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _frame(self, time: float):
        from worldsolver.contracts import StateFrame

        position = np.ascontiguousarray(self.x[: self.n_surface].copy())
        velocity = np.ascontiguousarray(self.v[: self.n_surface].copy())
        position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

        mass = self.export_mass.copy()
        gap = position[:, 2] - self.ground_z
        external_force = mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=np.zeros(self.n_surface, dtype=np.int64),
            attributes={
                "mass_quantity": mass.copy(),
                "deformation_gradient": self._node_deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external_force,
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    # --------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        target = float(time)
        target_step = int(round(target / self.dt))
        guard = 0
        max_steps = self.substeps * (len(self.sample_times) + 4) + 16
        while self.step_index < target_step and guard <= max_steps:
            self._step()
            guard += 1
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict:
        diag = dict(self._diag)
        diag["steps_taken"] = int(self.step_index)
        diag["final_time"] = float(self.time)
        diag["mean_plastic_strain"] = float(np.mean(self.plastic_strain))
        diag["max_plastic_strain"] = float(np.max(self.plastic_strain))
        diag["min_height_m"] = float(np.min(self.x[: self.n_surface, 2]))
        diag["max_height_m"] = float(np.max(self.x[: self.n_surface, 2]))
        return diag
