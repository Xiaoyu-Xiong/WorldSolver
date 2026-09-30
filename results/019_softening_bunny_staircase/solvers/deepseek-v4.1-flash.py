"""Softening viscoplastic bunny on a staircase.

Implements a deterministic, self-contained large-deformation solid solver
without any external physics engine.  The bunny surface mesh is treated as a
meshfree / spring-lattice continuum solid:

* nonlinear edge springs hold the shape (membrane resistance),
* dihedral (bending) springs resist folding,
* an internal-pressure term supplies bulk / volume resistance,
* gravity, viscous damping and Coulomb contact against the frozen boxes.

Constitutive behaviour is elasto-viscoplastic with work softening: the
per-edge rest length flows when the strain exceeds a yield threshold, the
flow is rate limited by a plastic viscosity (Perzyna overstress), accumulated
plastic strain softens the elastic stiffness and lowers the yield threshold.
Permanent deformation, softened-region sag and volume change therefore emerge
from the integration rather than from scripted animation.

Outputs every required StateFrame field: per-vertex deformation gradient
(area weighted triangle deformation gradient in material coordinates),
per-vertex mass, contact signed distance, cumulative mass source, paired
interaction forces per obstacle and the frozen contact length scale.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


class BenchmarkSolver:
    """Viscoplastic mass-spring / internal-pressure deformable solid."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config

        rest = np.asarray(scene.vertices, dtype=float).copy()
        self.rest = rest
        self.n_verts = int(rest.shape[0])
        self.faces = self._load_faces(config)

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.total_mass = float(config.get("subject_mass", 2.4))
        self.mass = np.full(self.n_verts, self.total_mass / self.n_verts)
        self.inv_mass = 1.0 / self.mass

        self.stiffness = float(config.get("stiffness", 22.0))
        self.softening_rate = float(config.get("softening_rate", 0.34))
        self.damping = float(config.get("damping", 2.3))
        self.restitution = float(config.get("restitution", 0.12))
        self.friction = float(config.get("friction", 0.72))
        self.contact_length_scale = float(config.get("contact_length_scale", 0.04))
        self.radius = 0.5 * self.contact_length_scale
        self.duration = float(config.get("duration", 4.2))
        self.sample_dt = float(config.get("sample_dt", self.duration / 126.0))

        # -- lattice geometry -------------------------------------------------
        faces = self.faces
        if faces.size:
            edges = np.concatenate(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
            )
            edges = np.unique(np.sort(edges, axis=1), axis=0)
        else:
            edges = self._knn_edges(rest, 6)
        self.edges = edges
        self.e_i = edges[:, 0].astype(np.int64)
        self.e_j = edges[:, 1].astype(np.int64)
        self.rest_len = np.maximum(
            np.linalg.norm(rest[self.e_j] - rest[self.e_i], axis=1), 1e-6
        )
        self.plastic_len = self.rest_len.copy()
        self.plastic_strain = np.zeros(len(edges))
        self.soft = np.ones(len(edges))

        self.bend = self._bending_pairs(faces)
        if self.bend.size:
            self.b_i = self.bend[:, 0].astype(np.int64)
            self.b_j = self.bend[:, 1].astype(np.int64)
            self.rest_bend = np.maximum(
                np.linalg.norm(rest[self.b_j] - rest[self.b_i], axis=1), 1e-6
            )
        else:
            self.b_i = self.b_j = np.zeros(0, dtype=np.int64)
            self.rest_bend = np.zeros(0)

        # -- triangle data for pressure and deformation-gradient diagnostics --
        self.t0 = faces[:, 0].astype(np.int64) if faces.size else np.zeros(0, np.int64)
        self.t1 = faces[:, 1].astype(np.int64) if faces.size else np.zeros(0, np.int64)
        self.t2 = faces[:, 2].astype(np.int64) if faces.size else np.zeros(0, np.int64)
        self.n_tri = int(len(self.t0))
        if self.n_tri:
            self.tri_area, self.Q_rest, self.D0_inv = self._rest_tri_frames()
            cr = np.cross(rest[self.t1] - rest[self.t0], rest[self.t2] - rest[self.t0])
            self.vol0 = max(
                float(np.einsum("ij,ij->i", rest[self.t0], cr).sum() / 6.0), 1e-6
            )
        else:
            self.tri_area = np.zeros(0)
            self.vol0 = 1.0

        # -- constitutive / integration parameters ----------------------------
        self.k_edge = 0.4 * self.stiffness
        self.k_bend = 0.3 * self.k_edge
        self.comp_stiff = 3.0
        self.c_damp = 0.002 * self.damping * self.k_edge
        self.gamma = 0.1 * self.damping
        self.bulk = 6.0 * self.stiffness

        self.yield0 = 0.22
        self.alpha_ref = 0.5
        self.plastic_visc = 0.10

        self.k_pen = 3000.0
        self.c_pen = 2.0
        self.c_fric = 0.5
        self.projection_depth = 0.006

        n_sub = 100
        self.dt_sub = self.sample_dt / n_sub

        # -- contact obstacles (frozen scene geometry) ------------------------
        self.box_c, self.box_h, self.box_ids = self._load_boxes(config)
        self.n_boxes = len(self.box_c)
        self._jbox = np.zeros((self.n_boxes, 3))

        # -- state ------------------------------------------------------------
        self.pos = rest.copy()
        vel0 = np.asarray(config.get("initial_velocity", [0.0, 0.0, 0.0]), float)
        self.vel = np.tile(vel0, (self.n_verts, 1))
        self.force = np.zeros_like(self.pos)
        self.step_index = 0
        self.body_id = np.zeros(self.n_verts, dtype=np.int64)
        self.min_volume_ratio = 1.0
        self.max_abs_strain = 0.0

    # --------------------------------------------------------------- geometry
    def _asset_path(self, config) -> Path | None:
        base = Path(__file__).resolve().parent.parent
        value = config.get("subject_asset")
        if value:
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = base / candidate
            if candidate.is_file():
                return candidate
        fallback = base / "assets" / "subject.npz"
        return fallback if fallback.is_file() else None

    def _load_faces(self, config) -> np.ndarray:
        path = self._asset_path(config)
        if path is not None:
            try:
                with np.load(path) as data:
                    if "faces" in data.files:
                        faces = np.asarray(data["faces"], dtype=np.int64)
                        if faces.ndim == 2 and faces.shape[1] == 3:
                            return faces
            except Exception:
                pass
        try:  # defensive fallback only; the frozen asset provides the topology
            from scipy.spatial import ConvexHull

            hull = ConvexHull(self.rest)
            return np.asarray(hull.simplices, dtype=np.int64)
        except Exception:
            return np.zeros((0, 3), dtype=np.int64)

    @staticmethod
    def _knn_edges(rest: np.ndarray, k: int) -> np.ndarray:
        n = len(rest)
        k = min(k, n - 1)
        out = set()
        step = 512
        for start in range(0, n, step):
            block = rest[start : start + step]
            dist = np.linalg.norm(block[:, None, :] - rest[None, :, :], axis=2)
            idx = np.argsort(dist, axis=1)[:, 1 : k + 1]
            for row, neighbours in enumerate(idx):
                a = start + row
                for b in neighbours:
                    out.add((a, int(b)) if a < b else (int(b), a))
        return np.array(sorted(out), dtype=np.int64)

    @staticmethod
    def _bending_pairs(faces: np.ndarray) -> np.ndarray:
        if not faces.size:
            return np.zeros((0, 2), dtype=np.int64)
        owners: dict[tuple[int, int], list[int]] = defaultdict(list)
        for t, (a, b, c) in enumerate(faces):
            for u, v in ((a, b), (b, c), (c, a)):
                owners[(int(u), int(v)) if u < v else (int(v), int(u))].append(t)
        pairs = []
        for tris in owners.values():
            if len(tris) != 2:
                continue
            diff = np.setxor1d(faces[tris[0]], faces[tris[1]])
            if diff.size == 2:
                pairs.append((int(diff[0]), int(diff[1])))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def _rest_tri_frames(self):
        rest = self.rest
        p0, p1, p2 = rest[self.t0], rest[self.t1], rest[self.t2]
        e1 = p1 - p0
        e2 = p2 - p0
        cr = np.cross(e1, e2)
        area = np.maximum(0.5 * np.linalg.norm(cr, axis=1), 1e-12)
        n = cr / np.maximum(np.linalg.norm(cr, axis=1), 1e-12)[:, None]
        e1n = np.maximum(np.linalg.norm(e1, axis=1), 1e-12)
        u = e1 / e1n[:, None]
        w = n
        v = np.cross(w, u)
        Q = np.stack([u, v, w], axis=2)  # columns are the local basis vectors
        a11 = np.einsum("ij,ij->i", e1, u)
        a21 = np.einsum("ij,ij->i", e1, v)
        a12 = np.einsum("ij,ij->i", e2, u)
        a22 = np.einsum("ij,ij->i", e2, v)
        D0 = np.zeros((len(self.t0), 2, 2))
        D0[:, 0, 0] = a11
        D0[:, 1, 0] = a21
        D0[:, 0, 1] = a12
        D0[:, 1, 1] = a22
        try:
            D0_inv = np.linalg.inv(D0)
        except np.linalg.LinAlgError:
            D0_inv = np.tile(np.eye(2), (len(self.t0), 1, 1))
        bad = ~np.isfinite(D0_inv).all(axis=(1, 2))
        if bad.any():
            D0_inv[bad] = np.eye(2)
        return area, Q, D0_inv

    @staticmethod
    def _load_boxes(config):
        objects = (config.get("scene_geometry") or {}).get("objects") or []
        centres, halves, names = [], [], []
        for obj in objects:
            geom = obj.get("geometry") or {}
            if geom.get("type") != "box" or not obj.get("collision_enabled", True):
                continue
            he = geom.get("half_extents")
            if not he or len(he) != 3:
                continue
            tr = (obj.get("transform") or {}).get("translation") or [0.0, 0.0, 0.0]
            centres.append([float(v) for v in tr])
            halves.append([float(v) for v in he])
            names.append(str(obj.get("id", f"box{len(names)}")))
        if not centres:  # frozen defaults from scene_geometry.yaml
            default = [
                ("ground", [0.0, 0.0, -0.04], [3.0, 1.6, 0.04]),
                ("stair_0", [-0.75, 0.0, 0.96], [0.36, 1.05, 0.14]),
                ("stair_1", [-0.15, 0.0, 0.68], [0.36, 1.05, 0.14]),
                ("stair_2", [0.45, 0.0, 0.40], [0.36, 1.05, 0.14]),
                ("stair_3", [1.05, 0.0, 0.12], [0.36, 1.05, 0.14]),
            ]
            for name, c, h in default:
                names.append(name)
                centres.append(c)
                halves.append(h)
        return (
            np.asarray(centres, dtype=float),
            np.asarray(halves, dtype=float),
            tuple(names),
        )

    # ------------------------------------------------------------- contact
    def _contact_query(self, pos: np.ndarray):
        rel = pos[:, None, :] - self.box_c[None, :, :]
        q = np.abs(rel) - self.box_h[None, :, :]
        qp = np.maximum(q, 0.0)
        d_out = np.linalg.norm(qp, axis=2)
        d_in = np.minimum(np.max(q, axis=2), 0.0)
        sd = d_out + d_in

        sgn = np.where(rel >= 0.0, 1.0, -1.0)
        outer = qp * sgn
        n_out = outer / np.maximum(d_out, _EPS)[:, :, None]
        k = np.argmax(q, axis=2)
        n_in = np.zeros_like(rel)
        axis_sign = np.take_along_axis(sgn, k[:, :, None], axis=2)
        np.put_along_axis(n_in, k[:, :, None], axis_sign, axis=2)
        outside = (d_out > 0.0)[:, :, None]
        normals = np.where(outside, n_out, n_in)

        sd_eff = sd - self.radius
        idx = np.argmin(sd_eff, axis=1)
        rows = np.arange(pos.shape[0])
        return sd_eff[rows, idx], normals[rows, idx], idx

    def _apply_contact(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> None:
        sd, nrm, idx = self._contact_query(pos)
        pen = np.maximum(-sd, 0.0)
        if not np.any(pen > 0.0):
            return
        vn = np.einsum("ij,ij->i", vel, nrm)
        f_n = self.k_pen * pen - self.c_pen * np.minimum(vn, 0.0)
        f_n = np.maximum(f_n, 0.0)
        vt = vel - vn[:, None] * nrm
        vt_mag = np.linalg.norm(vt, axis=1)
        f_t = np.minimum(self.c_fric * vt_mag, self.friction * f_n)
        fvec = f_n[:, None] * nrm - (f_t / np.maximum(vt_mag, _EPS))[:, None] * vt
        np.add.at(self._jbox, idx, fvec * dt)

        deep = sd < -self.projection_depth
        if np.any(deep):
            pos[deep] -= sd[deep, None] * nrm[deep]

    # ---------------------------------------------------------- constitutive
    def _plastic_update(self, dt: float) -> None:
        d = self.pos[self.e_j] - self.pos[self.e_i]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        strain = (length - self.plastic_len) / self.plastic_len
        soft = np.maximum(
            1.0 - self.softening_rate * np.minimum(self.plastic_strain / self.alpha_ref, 1.0),
            0.25,
        )
        self.soft = soft
        excess = np.abs(strain) - self.yield0 * soft
        rate = np.where(excess > 0.0, excess / self.plastic_visc * np.sign(strain), 0.0)
        d_l0 = np.clip(dt * rate * self.plastic_len, -0.05 * self.rest_len, 0.05 * self.rest_len)
        self.plastic_len += d_l0
        np.clip(self.plastic_len, 0.25 * self.rest_len, 4.0 * self.rest_len, out=self.plastic_len)
        self.plastic_strain += dt * np.abs(rate)
        if strain.size:
            self.max_abs_strain = max(self.max_abs_strain, float(np.abs(strain).max()))

    # ------------------------------------------------------------- dynamics
    def _substep(self, dt: float) -> None:
        pos, vel = self.pos, self.vel
        f = self.mass[:, None] * self.gravity[None, :]

        ei, ej = self.e_i, self.e_j
        d = pos[ej] - pos[ei]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        dirv = d / length[:, None]
        stretch = length - self.plastic_len
        k = self.k_edge * self.soft
        k = np.where(stretch < 0.0, k * self.comp_stiff, k)
        ldot = np.einsum("ij,ij->i", vel[ej] - vel[ei], dirv)
        fs = k * stretch + self.c_damp * ldot
        fv = fs[:, None] * dirv
        np.add.at(f, ei, fv)
        np.add.at(f, ej, -fv)

        if self.b_j.size:
            bi, bj = self.b_i, self.b_j
            db = pos[bj] - pos[bi]
            lb = np.maximum(np.linalg.norm(db, axis=1), 1e-9)
            dirb = db / lb[:, None]
            ldotb = np.einsum("ij,ij->i", vel[bj] - vel[bi], dirb)
            fb = self.k_bend * (lb - self.rest_bend) + self.c_damp * ldotb
            fbv = fb[:, None] * dirb
            np.add.at(f, bi, fbv)
            np.add.at(f, bj, -fbv)

        if self.n_tri:
            d1 = pos[self.t1] - pos[self.t0]
            d2 = pos[self.t2] - pos[self.t0]
            cr = np.cross(d1, d2)
            vol = float(np.einsum("ij,ij->i", pos[self.t0], cr).sum() / 6.0)
            gv = np.zeros_like(pos)
            np.add.at(gv, self.t0, cr / 6.0)
            np.add.at(gv, self.t1, cr / 6.0)
            np.add.at(gv, self.t2, cr / 6.0)
            pressure = self.bulk * (self.vol0 - vol) / self.vol0
            f += pressure * gv
            self.min_volume_ratio = min(self.min_volume_ratio, vol / self.vol0)

        f -= self.gamma * self.mass[:, None] * vel
        vel += dt * f * self.inv_mass[:, None]
        pos += dt * vel

        self._apply_contact(pos, vel, dt)
        # deterministic numerical safety net (never reached by the physical motion)
        np.clip(vel, -50.0, 50.0, out=vel)
        np.clip(pos, -50.0, 50.0, out=pos)
        self._plastic_update(dt)

    # ------------------------------------------------------------ trajectory
    def _deformation_gradient(self) -> np.ndarray:
        n = self.n_verts
        if not self.n_tri:
            return np.tile(np.eye(3), (n, 1, 1))
        pos = self.pos
        p0, p1, p2 = pos[self.t0], pos[self.t1], pos[self.t2]
        d1 = p1 - p0
        d2 = p2 - p0
        cr = np.cross(d1, d2)
        cn = np.linalg.norm(cr, axis=1)
        bad = cn < 1e-14
        nrm = cr / np.where(bad, 1.0, cn)[:, None]
        if bad.any():
            nrm[bad] = self.Q_rest[bad][:, :, 2]
        l1 = np.maximum(np.linalg.norm(d1, axis=1), 1e-12)
        u = d1 / l1[:, None]
        v = np.cross(nrm, u)
        Qc = np.stack([u, v, nrm], axis=2)

        D = np.zeros((self.n_tri, 2, 2))
        D[:, 0, 0] = l1
        D[:, 1, 0] = np.einsum("ij,ij->i", d1, v)
        D[:, 0, 1] = np.einsum("ij,ij->i", d2, u)
        D[:, 1, 1] = np.einsum("ij,ij->i", d2, v)
        g = D @ self.D0_inv
        M = np.zeros((self.n_tri, 3, 3))
        M[:, 0, 0] = g[:, 0, 0]
        M[:, 0, 1] = g[:, 0, 1]
        M[:, 1, 0] = g[:, 1, 0]
        M[:, 1, 1] = g[:, 1, 1]
        M[:, 2, 2] = 1.0
        if bad.any():
            M[bad] = np.eye(3)
        Ft = np.einsum("tij,tjk,tlk->til", Qc, M, self.Q_rest)

        num = np.zeros((n, 3, 3))
        den = np.zeros(n)
        wF = self.tri_area[:, None, None] * Ft
        for corner in (self.t0, self.t1, self.t2):
            np.add.at(num, corner, wF)
            np.add.at(den, corner, self.tri_area)
        den = np.maximum(den, _EPS)
        F = num / den[:, None, None]
        if not np.isfinite(F).all():
            F = np.tile(np.eye(3), (n, 1, 1))
        return F

    def advance_to(self, time: float):
        time = float(time)
        target = int(round(time / self.dt_sub))
        if target < self.step_index:
            target = self.step_index
        start = self.step_index
        for _ in range(target - start):
            self._substep(self.dt_sub)
            self.step_index += 1
        nsteps = target - start
        if nsteps > 0:
            mean_force = self._jbox / (nsteps * self.dt_sub)
        else:
            mean_force = np.zeros_like(self._jbox)
        self._jbox = np.zeros_like(self._jbox)

        sd, _, _ = self._contact_query(self.pos)
        force = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": sd,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "interaction_force": mean_force.copy(),
                "external_force": force,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "viscoplastic_lattice_internal_pressure",
            "vertices": int(self.n_verts),
            "edges": int(len(self.e_i)),
            "bending_pairs": int(len(self.b_i)),
            "triangles": int(self.n_tri),
            "steps": int(self.step_index),
            "substep_dt": float(self.dt_sub),
            "min_volume_ratio": float(self.min_volume_ratio),
            "max_abs_strain": float(self.max_abs_strain),
            "max_plastic_strain": float(self.plastic_strain.max()) if len(self.plastic_strain) else 0.0,
            "mean_stiffness_factor": float(self.soft.mean()) if len(self.soft) else 1.0,
        }
