"""Physically-based solver for 066_octopus_cylinder_squash.

Volumetric elastic octopus (Stable Neo-Hookean flesh):
  * 34 simulation nodes = the frozen rest-surface vertices (renderer binding).
  * Mantle: Delaunay tetrahedral FEM with a compressible neo-Hookean energy
    (hyperelastic, near-incompressible) -> true 3D solid response.
  * Arms + surface: axial springs on every mesh edge and bending springs on
    graph-distance-2 pairs, giving elastic tentacles that drape and recover.
  * Rigid cylinders (3 static supports + 1 kinematic upper): penalty contact
    with normal damping and Coulomb friction; the upper cylinder follows the
    frozen piecewise-linear keyframes.
  * Implicit Euler with Newton iterations, exact linear drag, substepped.
All required raw trajectory fields are exported each sample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
TOTAL_MASS = 12.0
N_NODES = 34

# material (flesh-like, near incompressible)
MU = 1800.0
LAMBDA = 12000.0
# springs
K_EDGE = 2000.0      # axial, all mesh edges
K_BEND = 60.0        # bending, distance-2 pairs
C_SPRING = 1.5       # axial spring damping
C_DRAG = 1.5         # global drag per unit mass (1/s)
# contact
K_CONTACT = 4.0e4
C_CONTACT = 250.0
MU_FRICTION = 0.65
V_EPS = 0.05
F_CAP = 4.0e4        # per-node contact force cap
FLOOR_Z = 0.0
K_FLOOR = 8.0e3

N_SUB = 10           # substeps per 0.02 s sample
NEWTON_MAX = 6
V_CLAMP = 60.0


def _parse_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.asarray(verts, float), np.asarray(faces, int)


def _quat_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class BenchmarkSolver:
    # ------------------------------------------------------------ lifecycle
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        base = Path(__file__).resolve().parent.parent

        self.rest, faces = _parse_obj(base / "assets" / "octopus.obj")
        if self.rest.shape[0] != N_NODES:
            N = self.rest.shape[0]
        else:
            N = N_NODES
        self.n_nodes = N

        self.duration = float(cfg.get("duration", 8.0))
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.gravity_vec = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)

        # kinematic upper cylinder keyframes (frozen config)
        motion = cfg.get("upper_cylinder_motion", {})
        kf = motion.get("keyframes", [])
        if kf:
            self.kf_t = np.array([float(k["time"]) for k in kf])
            self.kf_z = np.array([float(k["translation"][2]) for k in kf])
        else:
            self.kf_t = np.array([0.0, 8.0])
            self.kf_z = np.array([4.2, 4.2])

        # rigid cylinders from frozen scene geometry
        self.cyls = self._load_cylinders(base)

        # masses
        self.mass = np.full(N, TOTAL_MASS / N)
        self.inv_m = 1.0 / self.mass

        # ---- tet elements: mantle = convex hull of the first 10 vertices
        self._build_tets()
        # ---- spring elements: mesh edges + distance-2 bending pairs
        self._build_springs(faces)

        # ---- state
        self.x = self.rest.copy()
        self.v = np.zeros_like(self.x)
        self.t = 0.0
        self.impulse = np.zeros(3)          # cumulative external impulse
        self.diag = {"steps": 0, "newton_fail": 0, "n_tet": self.n_tet,
                     "n_spring": self.n_spr}

        # ---- frozen output scaffolding
        self.domain_w = np.ones((1, N))
        self.mass_quantity = self.mass.reshape(N, 1).copy()
        self.zero_g3 = np.zeros((1, 3))
        self.mass_source = np.zeros(1)
        speed = np.linalg.norm(self.gravity_vec)
        self.ref_momentum = float(TOTAL_MASS * np.sqrt(speed * 1.0))

    def _load_cylinders(self, base: Path):
        text = (base / "scene_geometry.yaml").read_text(encoding="utf-8")
        try:
            import yaml
            geom = yaml.safe_load(text)
        except Exception:
            from worldsolver.config import load_yaml
            geom = load_yaml(base / "scene_geometry.yaml")
        cyls = []
        for obj in geom["objects"]:
            if obj["geometry"]["type"] != "cylinder":
                continue
            tr = obj["transform"]
            axis = _quat_rot(np.asarray(tr["rotation_xyzw"], float)) @ np.array([0.0, 0.0, 1.0])
            cyls.append({
                "id": obj["id"],
                "center": np.asarray(tr["translation"], float),
                "axis": axis / np.linalg.norm(axis),
                "radius": float(obj["geometry"]["radius"]),
                "half_len": float(obj["geometry"]["height"]) / 2.0,
                "kinematic": obj["motion"] == "kinematic",
            })
        return cyls

    def _build_tets(self):
        n_mantle = min(10, self.n_nodes)
        mantle_idx = np.arange(n_mantle)
        p = self.rest[mantle_idx]
        d = Delaunay(p)
        tets, dm_inv, vol0 = [], [], []
        for s in d.simplices:
            idx = mantle_idx[s]
            q = self.rest[idx]
            dm = (q[1:] - q[0]).T              # columns = edge vectors
            det = np.linalg.det(dm)
            if abs(det) < 1e-9:
                continue
            tets.append(idx)
            dm_inv.append(np.linalg.inv(dm))
            vol0.append(abs(det) / 6.0)
        self.tet_idx = np.asarray(tets, int) if tets else np.zeros((0, 4), int)
        self.dm_inv = np.asarray(dm_inv) if tets else np.zeros((0, 3, 3))
        self.vol0 = np.asarray(vol0) if tets else np.zeros(0)
        self.n_tet = len(self.tet_idx)
        if self.n_tet:
            self.tet_dof = (self.tet_idx[:, :, None] * 3
                            + np.arange(3)[None, None, :]).reshape(self.n_tet, 12)
        else:
            self.tet_dof = np.zeros((0, 12), int)

    def _build_springs(self, faces):
        edges = set()
        for tri in faces:
            m = len(tri)
            for j in range(m):
                a, b = int(tri[j]), int(tri[(j + 1) % m])
                edges.add((min(a, b), max(a, b)))
        adj = {i: set() for i in range(self.n_nodes)}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        pairs, ks = [], []
        for a, b in sorted(edges):
            pairs.append((a, b))
            ks.append(K_EDGE)
        bend = set()
        for i in range(self.n_nodes):
            for a in adj[i]:
                for b in adj[i]:
                    if a < b and (a, b) not in edges:
                        bend.add((a, b))
        for a, b in sorted(bend):
            pairs.append((a, b))
            ks.append(K_BEND)
        self.spring_pairs = np.asarray(pairs, int) if pairs else np.zeros((0, 2), int)
        self.spring_k = np.asarray(ks, float)
        self.n_spr = len(self.spring_pairs)
        if self.n_spr:
            d0 = np.linalg.norm(self.rest[self.spring_pairs[:, 0]]
                                - self.rest[self.spring_pairs[:, 1]], axis=1)
            self.spring_l0 = np.maximum(d0, 1e-6)
        else:
            self.spring_l0 = np.zeros(0)

    # ------------------------------------------------------------ kinematics
    def _upper_z(self, t):
        return float(np.interp(t, self.kf_t, self.kf_z))

    def _upper_vz(self, t):
        h = 1e-4
        return (self._upper_z(min(t + h, self.duration))
                - self._upper_z(max(t - h, 0.0))) / (2 * h)

    def _cyl_states(self, t):
        out = []
        zc = self._upper_z(t)
        vz = self._upper_vz(t)
        for c in self.cyls:
            if c["kinematic"]:
                ctr = c["center"].copy()
                ctr[2] = zc
                vel = np.array([0.0, 0.0, vz])
            else:
                ctr = c["center"]
                vel = np.zeros(3)
            out.append((ctr, c["axis"], c["radius"], c["half_len"], vel))
        return out

    # ------------------------------------------------------------ forces
    def _tet_node_forces(self, x):
        """Per-tet nodal forces, shape [Ne,4,3] (hyperelastic neo-Hookean)."""
        if self.n_tet == 0:
            return np.zeros((0, 4, 3))
        idx = self.tet_idx
        p0 = x[idx[:, 0]]
        H = np.stack([x[idx[:, k + 1]] - p0 for k in range(3)], axis=2)
        F = H @ self.dm_inv
        J = np.linalg.det(F)
        Js = np.maximum(np.abs(J), 1e-4) * np.where(J < 0, -1.0, 1.0)
        Finv = np.linalg.inv(F)
        FinvT = np.swapaxes(Finv, 1, 2)
        P = MU * (F - FinvT) + LAMBDA * np.log(Js)[:, None, None] * FinvT
        G = self.vol0[:, None, None] * (P @ np.swapaxes(self.dm_inv, 1, 2))
        f = np.empty((self.n_tet, 4, 3))
        f[:, 1:, :] = -np.swapaxes(G, 1, 2)   # -dE/dH columns
        f[:, 0, :] = G.sum(axis=2)
        return f

    def _tet_forces(self, x):
        f = np.zeros_like(x)
        if self.n_tet:
            np.add.at(f, self.tet_idx.reshape(-1), self._tet_node_forces(x).reshape(-1, 3))
        return f

    def _spring_forces(self, x, v):
        f = np.zeros_like(x)
        if self.n_spr == 0:
            return f
        pa, pb = self.spring_pairs[:, 0], self.spring_pairs[:, 1]
        d = x[pb] - x[pa]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        n = d / L[:, None]
        dv = v[pb] - v[pa]
        s = self.spring_k * (L - self.spring_l0) \
            + C_SPRING * np.einsum("ij,ij->i", dv, n)
        fs = s[:, None] * n
        np.add.at(f, pb, -fs)
        np.add.at(f, pa, fs)
        return f

    def _contact_forces(self, x, v, t):
        f = np.zeros_like(x)
        for ctr, axis, R, hl, cv in self._cyl_states(t):
            rel = x - ctr
            along = rel @ axis
            radial = rel - along[:, None] * axis[None, :]
            r = np.linalg.norm(radial, axis=1)
            rs = np.maximum(r, 1e-9)
            act = (r < R) & (np.abs(along) <= hl)
            if not act.any():
                continue
            nrm = radial / rs[:, None]
            pen = R - r
            vn = np.einsum("ij,ij->i", v - cv[None, :], nrm)
            fn = K_CONTACT * pen + C_CONTACT * np.maximum(-vn, 0.0)
            vt = (v - cv[None, :]) - vn[:, None] * nrm
            sp = np.linalg.norm(vt, axis=1)
            tot = fn[:, None] * nrm
            sel = act & (sp > 1e-12)
            if sel.any():
                mag = MU_FRICTION * np.maximum(fn[sel], 0.0) * np.tanh(sp[sel] / V_EPS)
                tot[sel] -= mag[:, None] * vt[sel] / sp[sel, None]
            tot[~act] = 0.0
            mag = np.linalg.norm(tot, axis=1)
            scale = np.where(mag > F_CAP, F_CAP / np.maximum(mag, 1e-12), 1.0)
            f += tot * scale[:, None]
        below = x[:, 2] < FLOOR_Z
        if below.any():
            vz = v[below, 2]
            f[below, 2] += K_FLOOR * (FLOOR_Z - x[below, 2]) + 50.0 * np.maximum(-vz, 0.0)
        return f

    def _external_forces(self, x, v, t):
        f = self.mass[:, None] * self.gravity_vec[None, :]
        return f + self._contact_forces(x, v, t)

    def _total_force(self, x, v_prev, t):
        f = self._tet_forces(x)
        f += self._spring_forces(x, v_prev)
        f += self._external_forces(x, v_prev, t)
        return f

    # ------------------------------------------------------------ jacobian
    def _assemble_K(self, x, v_prev, t):
        n = 3 * self.n_nodes
        K = np.zeros((n, n))
        # --- springs (analytic)
        if self.n_spr:
            pa, pb = self.spring_pairs[:, 0], self.spring_pairs[:, 1]
            d = x[pb] - x[pa]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            nn = d / L[:, None]
            s = self.spring_k * (L - self.spring_l0)
            stretch = s / L
            eye = np.eye(3)
            # f_b = -s*n ; df_b/dx_b = -(stretch*I + (k-stretch) n n^T)
            blk = -(stretch[:, None, None] * eye[None]
                    + (self.spring_k - stretch)[:, None, None]
                    * (nn[:, :, None] * nn[:, None, :]))
            for si in range(self.n_spr):
                a, b = pa[si], pb[si]
                ia, ib = slice(a * 3, a * 3 + 3), slice(b * 3, b * 3 + 3)
                K[ib, ib] += blk[si]
                K[ia, ia] += blk[si]
                K[ib, ia] -= blk[si]
                K[ia, ib] -= blk[si]
        # --- tets (numeric 12x12 blocks, central differences)
        if self.n_tet:
            h = 1e-6
            f0 = self._tet_node_forces(x)                # [Ne,4,3]
            for dof in range(12):
                node, comp = divmod(dof, 3)
                e = np.zeros(3)
                e[comp] = h
                xp = x.copy()
                xp[self.tet_idx[:, node]] += e
                fp = self._tet_node_forces(xp)
                xm = x.copy()
                xm[self.tet_idx[:, node]] -= e
                fm = self._tet_node_forces(xm)
                dK = ((fp - fm) / (2 * h)).reshape(self.n_tet, 12)
                rows = np.repeat((self.tet_idx[:, node] * 3 + comp)[:, None], 12,
                                 axis=1).reshape(-1)
                cols = self.tet_dof.reshape(-1)
                np.add.at(K, (rows, cols), dK.reshape(-1))
        return K

    # ------------------------------------------------------------ stepping
    def _step(self, dt, t0):
        x0, v0 = self.x, self.v
        tmid = t0 + 0.5 * dt
        inv_dt2 = 1.0 / (dt * dt)
        n = 3 * self.n_nodes
        x = x0 + dt * v0
        eye = np.eye(n)
        Md = np.repeat(self.inv_m, 3)
        v_prev = v0
        converged = False
        r0 = None
        for _ in range(NEWTON_MAX):
            v_est = (x - x0) / dt
            f = self._total_force(x, v_prev, tmid)
            r = ((x - x0) * inv_dt2 - self.inv_m[:, None] * f).reshape(-1) \
                + C_DRAG * v_est.reshape(-1)
            rnorm = float(np.linalg.norm(r))
            if r0 is None:
                r0 = max(rnorm, 1e-12)
            if rnorm <= 1e-6 * r0 or rnorm <= 1e-7:
                converged = True
                break
            K = self._assemble_K(x, v_prev, tmid)
            A = inv_dt2 * eye - Md[:, None] * K + (C_DRAG / dt) * eye
            try:
                dx = np.linalg.solve(A, -r)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(A, -r, rcond=None)[0]
            step_norm = np.abs(dx).max()
            if step_norm > 0.05:
                dx *= 0.05 / step_norm
            x = x + dx.reshape(self.n_nodes, 3)
            if np.linalg.norm(dx) < 1e-9:
                converged = True
                break
        if not converged:
            self.diag["newton_fail"] += 1
        v = (x - x0) / dt
        sp = np.linalg.norm(v, axis=1)
        over = sp > V_CLAMP
        if over.any():
            v[over] *= (V_CLAMP / sp[over])[:, None]
            x = x0 + dt * v
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            x, v = x0.copy(), np.zeros_like(v0)
        # cumulative external impulse (trapezoid over the substep)
        f0 = self._external_forces(x0, v0, t0).sum(axis=0)
        f1 = self._external_forces(x, v, t0 + dt).sum(axis=0)
        self.impulse += 0.5 * (f0 + f1) * dt
        self.x, self.v = x, v
        self.diag["steps"] += 1

    def _deformation_gradients(self):
        if self.n_tet == 0:
            return np.zeros((0, 3, 3))
        idx = self.tet_idx
        p0 = self.x[idx[:, 0]]
        H = np.stack([self.x[idx[:, k + 1]] - p0 for k in range(3)], axis=2)
        return H @ self.dm_inv

    # ------------------------------------------------------------ interface
    def advance_to(self, time: float):
        target = float(time)
        dt_full = self.sample_dt / N_SUB
        max_steps = int(round(self.duration / self.sample_dt)) * N_SUB + 64
        guard = 0
        while self.t < target - 1e-12 and guard < max_steps:
            dt = min(dt_full, target - self.t)
            self._step(dt, self.t)
            self.t += dt
            guard += 1
        f_ext = self._external_forces(self.x, self.v, self.t).sum(axis=0)
        F = self._deformation_gradients()
        frame = StateFrame(
            time=self.t,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n_nodes, dtype=np.int32),
            attributes={
                "rest_position": self.rest.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "budget_domain_weights": self.domain_w.copy(),
                "budget_external_force": f_ext.reshape(1, 1, 3),
                "budget_cumulative_external_impulse": self.impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": self.zero_g3.copy(),
                "budget_cumulative_outflow_momentum": self.zero_g3.copy(),
                "cumulative_mass_source": self.mass_source.copy(),
                "element_deformation_gradient": F.copy(),
            },
            scalars={
                "reference_momentum": self.ref_momentum,
                "gravity_z": float(self.gravity_vec[2]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        d = dict(self.diag)
        d["time"] = self.t
        if self.n_tet:
            d["min_detF"] = float(np.linalg.det(self._deformation_gradients()).min())
        return d
