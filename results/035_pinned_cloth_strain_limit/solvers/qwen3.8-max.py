"""PBD cloth solver for the Figure 15(c) sphere-drape strain-limit experiment.

A mass-spring / position-based-dynamics cloth with strict edge-strain range
constraints (principal stretches kept inside [0.9, 1.1] via a safety margin),
weak bending resistance, and projected collision against the fixed sphere and
ground with Coulomb friction.  All required raw StateFrame fields are exported:
per-sample deformation gradient, mass quantity, contact signed distance,
cumulative mass source, and the frozen contact length scale.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()
        meta = scene.metadata()

        self._rng = np.random.default_rng(seed)

        gravity = np.asarray(cfg.get("gravity", meta.get("gravity", [0.0, 0.0, -9.81])), float)
        self.gravity = gravity

        cloth_cfg = cfg.get("cloth", {})
        meta_cloth = meta.get("cloth", {})
        self.thickness = float(cloth_cfg.get("thickness", meta_cloth.get("thickness", 0.012)))
        strain_lower = float(cloth_cfg.get("strain_lower", meta_cloth.get("strain_lower", 0.9)))
        strain_upper = float(cloth_cfg.get("strain_upper", meta_cloth.get("strain_upper", 1.1)))
        self.damping = float(cloth_cfg.get("damping", 1.35))
        self.friction = float(cloth_cfg.get("friction", 0.22))
        # Safety margin (40% toward unit stretch) inside the strict band [0.9, 1.1].
        self.strain_lo = strain_lower + 0.4 * (1.0 - strain_lower)
        self.strain_hi = strain_upper - 0.4 * (strain_upper - 1.0)

        sphere_cfg = cfg.get("sphere", {})
        meta_sphere = meta.get("sphere", {})
        self.sphere_center = np.asarray(
            sphere_cfg.get("center", meta_sphere.get("center", [0.0, 0.0, 0.64])), float
        )
        self.sphere_radius = float(sphere_cfg.get("radius", meta_sphere.get("radius", 0.64)))
        ground_cfg = cfg.get("ground", {})
        self.ground_z = float(
            ground_cfg.get("height", meta.get("ground_height", 0.0))
        )

        solver_cfg = cfg.get("solver", {})
        self.substep = float(solver_cfg.get("substep", 0.004))
        self.iterations = int(solver_cfg.get("constraint_iterations", 15))

        # ---------------- state ----------------
        self.n = int(frame0.position.shape[0])
        self.x = np.array(frame0.position, dtype=float, copy=True)
        self.v = np.array(frame0.velocity, dtype=float, copy=True)
        self.mass = np.array(frame0.mass, dtype=float, copy=True)
        self.inv_mass = np.where(self.mass > 0, 1.0 / np.maximum(self.mass, 1e-12), 0.0)
        self.x0 = np.array(self.x, copy=True)  # rest configuration
        self.t = 0.0

        attrs = dict(frame0.attributes)
        self.uv = np.array(attrs.get("uv", np.zeros((self.n, 2))), dtype=float)
        self.mass_quantity = np.array(self.mass, dtype=float, copy=True)
        self.contact_length_scale = float(
            frame0.scalars.get("contact_length_scale", self.thickness)
        )
        self.cumulative_mass_source = np.array(
            frame0.fields.get("cumulative_mass_source", np.zeros(1)), dtype=float
        ).ravel()

        # ---------------- topology ----------------
        faces = np.asarray(meta.get("topology", {}).get("faces"), dtype=int)
        if faces is None or faces.size == 0:  # pragma: no cover - fallback
            faces = self._grid_faces(meta_cloth.get("grid_shape", [31, 31]))
        self.faces = faces.reshape(-1, 3)
        self.mesh_edges = self._unique_edges(self.faces)
        self.bend_edges = self._bending_edges(self.faces, self.mesh_edges, self.n)

        e = self.mesh_edges
        d0 = self.x0[e[:, 0]] - self.x0[e[:, 1]]
        self.mesh_rest = np.linalg.norm(d0, axis=1)
        self.mesh_lo = self.strain_lo * self.mesh_rest
        self.mesh_hi = self.strain_hi * self.mesh_rest
        w = self.inv_mass[e[:, 0]] + self.inv_mass[e[:, 1]]
        self.mesh_wsum = np.where(w > 0, w, 1.0)

        b = self.bend_edges
        if len(b):
            d0b = self.x0[b[:, 0]] - self.x0[b[:, 1]]
            self.bend_rest = np.linalg.norm(d0b, axis=1)
            wb = self.inv_mass[b[:, 0]] + self.inv_mass[b[:, 1]]
            self.bend_wsum = np.where(wb > 0, wb, 1.0)
        else:  # pragma: no cover
            self.bend_rest = np.zeros(0)
            self.bend_wsum = np.zeros(0)
        self.bend_stiffness = 0.08

        # rest basis for the deformation gradient, F = S @ inv(M), identity at t=0
        p0 = self.x0[self.faces[:, 0]]
        p1 = self.x0[self.faces[:, 1]]
        p2 = self.x0[self.faces[:, 2]]
        e1 = p1 - p0
        e2 = p2 - p0
        nrm = np.cross(e1, e2)
        nlen = np.linalg.norm(nrm, axis=1)
        nhat = nrm / np.maximum(nlen, 1e-15)[:, None]
        self.rest_normal = nhat
        self.face_area = 0.5 * nlen
        M = np.stack([e1, e2, nhat], axis=2)  # (F,3,3)
        self.M_inv = np.linalg.inv(M)

        # vertex <- face averaging weights (uniform)
        self.vf_count = np.zeros(self.n)
        for k in range(3):
            np.add.at(self.vf_count, self.faces[:, k], 1.0)
        self.vf_inv = 1.0 / np.maximum(self.vf_count, 1.0)

        # collision radii
        self.sphere_r_eff = self.sphere_radius + 0.5 * self.thickness
        self.ground_z_eff = self.ground_z + 0.5 * self.thickness

        self._diag = {}

    @staticmethod
    def _grid_faces(grid_shape):  # pragma: no cover
        nr, nc = int(grid_shape[0]), int(grid_shape[1])
        faces = []
        for r in range(nr - 1):
            for c in range(nc - 1):
                i0 = r * nc + c
                i1 = i0 + 1
                i2 = i0 + nc
                i3 = i2 + 1
                faces.append((i0, i1, i2))
                faces.append((i1, i3, i2))
        return np.asarray(faces, dtype=int)

    @staticmethod
    def _unique_edges(faces):
        e = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        e = np.sort(e, axis=1)
        return np.unique(e, axis=0)

    @staticmethod
    def _bending_edges(faces, mesh_edges, n):
        key = {(int(min(a, b)), int(max(a, b))): [] for a, b in mesh_edges}
        for f in faces:
            a, b, c = (int(f[0]), int(f[1]), int(f[2]))
            for u, w in ((a, b), (b, c), (c, a)):
                key[(min(u, w), max(u, w))].append((a, b, c))
        bend = set()
        for (u, w), flist in key.items():
            if len(flist) == 2:
                t1, t2 = flist
                opp1 = next(k for k in t1 if k != u and k != w)
                opp2 = next(k for k in t2 if k != u and k != w)
                if opp1 != opp2:
                    bend.add((min(opp1, opp2), max(opp1, opp2)))
        if not bend:
            return np.zeros((0, 2), dtype=int)
        return np.asarray(sorted(bend), dtype=int)

    # -------------------------------------------------------------- stepping
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time > self.t:
            delta = time - self.t
            n_steps = max(1, int(round(delta / self.substep)))
            dt = delta / n_steps
            for _ in range(n_steps):
                self._step(dt)
            self.t = time
        return self._make_frame(self.t)

    def _step(self, dt: float) -> None:
        x_start = self.x
        # semi-implicit predictor
        self.v += self.gravity * dt
        p = self.x + self.v * dt

        mu = self.friction
        # friction bookkeeping: accumulated normal correction & tangent delta
        dn_acc = np.zeros(self.n)
        nrm_acc = np.zeros((self.n, 3))

        for _ in range(self.iterations):
            p = self._solve_edge_range(p, self.mesh_edges, self.mesh_rest,
                                     self.mesh_lo, self.mesh_hi, self.mesh_wsum, 1.0)
            if len(self.bend_edges):
                p = self._solve_edge_range(p, self.bend_edges, self.bend_rest,
                                           None, None, self.bend_wsum,
                                           self.bend_stiffness)
            p, dn, nn = self._project_collisions(p)
            dn_acc = np.maximum(dn_acc, dn)
            nrm_acc = np.where((dn > 0)[:, None], nn, nrm_acc)

        # Coulomb friction on the tangential motion of contacting vertices
        dp = p - x_start
        dpn = np.einsum("ij,ij->i", dp, nrm_acc)
        dpn = np.where(dn_acc > 0.0, dpn, 0.0)
        dp_n = dpn[:, None] * nrm_acc
        dp_t = dp - dp_n
        lt = np.linalg.norm(dp_t, axis=1)
        contacting = dn_acc > 1e-12
        scale = np.where(
            contacting & (lt > 1e-12),
            np.clip(1.0 - mu * dn_acc / np.maximum(lt, 1e-12), 0.0, 1.0),
            1.0,
        )
        p = x_start + dp_n + scale[:, None] * dp_t

        # final hard projection (guarantees no interpenetration)
        p, _, _ = self._project_collisions(p)

        self.x = p
        self.v = (self.x - x_start) / dt
        self.v *= float(np.exp(-self.damping * dt))
        speed = np.linalg.norm(self.v, axis=1)
        cap = np.where(speed > 25.0, 25.0 / np.maximum(speed, 1e-12), 1.0)
        self.v *= cap[:, None]

    def _solve_edge_range(self, p, edges, rest, lo, hi, wsum, stiff):
        if len(edges) == 0:
            return p
        i = edges[:, 0]
        j = edges[:, 1]
        d = p[i] - p[j]
        ln = np.linalg.norm(d, axis=1)
        if lo is None:
            target = rest
        else:
            target = np.where(ln > hi, hi, np.where(ln < lo, lo, ln))
        corr = stiff * (ln - target) / np.maximum(ln, 1e-12)
        wi = self.inv_mass[i]
        wj = self.inv_mass[j]
        c = (corr / wsum)[:, None] * d
        np.add.at(p, i, -wi[:, None] * c)
        np.add.at(p, j, wj[:, None] * c)
        return p

    def _project_collisions(self, p):
        dn = np.zeros(self.n)
        nn = np.zeros((self.n, 3))
        # sphere
        d = p - self.sphere_center
        r = np.linalg.norm(d, axis=1)
        pen_s = self.sphere_r_eff - r
        n_s = d / np.maximum(r, 1e-12)[:, None]
        # ground plane
        pen_g = self.ground_z_eff - p[:, 2]
        n_g = np.zeros((self.n, 3))
        n_g[:, 2] = 1.0
        # resolve the deeper penetration per vertex
        use_g = pen_g > pen_s
        pen = np.where(use_g, pen_g, pen_s)
        nh = np.where(use_g[:, None], n_g, n_s)
        hit = pen > 0.0
        if np.any(hit):
            p = p.copy()
            p[hit] += pen[hit, None] * nh[hit]
            dn[hit] = pen[hit]
            nn[hit] = nh[hit]
        return p, dn, nn

    # ---------------------------------------------------------------- output
    def _face_deformation_gradients(self):
        p = self.x
        f0 = self.faces[:, 0]
        f1 = self.faces[:, 1]
        f2 = self.faces[:, 2]
        e1 = p[f1] - p[f0]
        e2 = p[f2] - p[f0]
        n = np.cross(e1, e2)
        nl = np.linalg.norm(n, axis=1)
        nhat = np.where(
            (nl > 1e-15)[:, None], n / np.maximum(nl, 1e-15)[:, None], self.rest_normal
        )
        S = np.stack([e1, e2, nhat], axis=2)  # (F,3,3)
        return np.einsum("fij,fjk->fik", S, self.M_inv)

    def _vertex_deformation_gradients(self):
        ff = self._face_deformation_gradients()
        vf = np.zeros((self.n, 3, 3))
        for k in range(3):
            np.add.at(vf, self.faces[:, k], ff)
        return vf * self.vf_inv[:, None, None]

    def _signed_distances(self):
        d = self.x - self.sphere_center
        r = np.linalg.norm(d, axis=1)
        sd_sphere = r - self.sphere_r_eff
        sd_ground = self.x[:, 2] - self.ground_z_eff
        return np.minimum(sd_sphere, sd_ground)

    def _make_frame(self, time: float) -> StateFrame:
        sd = self._signed_distances()
        F = self._vertex_deformation_gradients()
        frame = StateFrame(
            float(time),
            np.array(self.x, copy=True),
            np.array(self.v, copy=True),
            mass=np.array(self.mass, copy=True),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": np.array(self.mass_quantity, copy=True),
                "uv": np.array(self.uv, copy=True),
            },
            fields={
                "contact_signed_distance": sd,
                "cumulative_mass_source": np.array(self.cumulative_mass_source, copy=True),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        self._diag = {
            "min_signed_distance": float(sd.min()),
            "num_contact": int(np.count_nonzero(sd < 1e-3)),
        }
        return frame

    def diagnostics(self) -> dict:
        return dict(self._diag)
