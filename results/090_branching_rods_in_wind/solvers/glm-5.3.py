"""Wind-driven branching elastic-rod solver (scene 090).

Physical model
--------------
The tree is discretized as a nodal network of the 49 rod vertices coupled by:

* near-inextensible stretch springs along every topology edge (tension/compression
  with mass-proportional stiffness and viscous damping),
* discrete bending joints at every interior vertex: for each pair of incident
  edges an angular spring resists deviation of the enclosed angle from its
  reference (source Figure 12) value, with viscous angular damping.  This makes
  the rods bending resistant; the shared junction vertices rigidly couple the
  positions (and, through the shared node frames, orientations) of the branches,
* prescribed wind load: a quadratic aerodynamic drag toward the local wind
  velocity, with the wind speed taken from the frozen keyframes, a
  height-and-branch-area-weighted spatial profile, and a per-branch gust phase
  offset so different branches respond with different amplitudes and timing,
* gravity and a small linear damping term.

Time integration is semi-implicit (symplectic) Euler with a fixed small
substep, root vertex pinned.  No Genesis, no keyframed animation: every frame
is produced by stepping the physics.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched quaternion product (xyzw order)."""
    av, aw = a[:, :3], a[:, 3:4]
    bv, bw = b[:, :3], b[:, 3:4]
    vec = np.cross(av, bv) + aw * bv + bw * av
    sca = aw * bw - np.einsum("ij,ij->i", av, bv)[:, None]
    return np.concatenate([vec, sca], axis=1)


class BenchmarkSolver:
    """Nodal discrete-rod solver for the branching tree in wind."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.base_attributes = {
            k: np.asarray(v).copy() for k, v in frame0.attributes.items()
        }

        # ---- topology --------------------------------------------------
        asset_rel = cfg["rod_network"]["topology_asset"]
        data = None
        for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
            path = base / asset_rel
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                break
        if data is None:
            raise FileNotFoundError(f"topology asset {asset_rel!r} not found")

        vertices = np.asarray(data["vertices"], dtype=float)
        edges = np.asarray(data["edges"], dtype=int)
        levels = np.asarray(data["vertex_level"], dtype=int)
        branch_id = np.asarray(data["branch_id"], dtype=int)
        lumped = np.asarray(data["lumped_length"], dtype=float)

        self.n = len(vertices)
        self.x = vertices.copy()
        self.x0 = vertices.copy()
        self.v = np.zeros((self.n, 3))
        self.q = np.zeros((self.n, 4))
        self.q[:, 3] = 1.0
        self.t = 0.0
        self.nsteps = 0

        # ---- masses ----------------------------------------------------
        m = np.asarray(frame0.mass, dtype=float).copy()
        junction_idx = np.asarray(cfg["rigid_junctions"]["vertex_indices"], dtype=int)
        mass_by_level = np.asarray(cfg["rigid_junctions"]["mass_by_level"], dtype=float)
        m[junction_idx] += mass_by_level[levels[junction_idx]]
        self.m = np.maximum(m, 1e-6)
        self.m_tot = float(self.m.sum())

        radius_by_level = np.asarray(cfg["rod_network"]["radius_by_level"], dtype=float)
        radii = radius_by_level[levels]
        self.levels = levels
        self.branch_id = branch_id

        # ---- edge (stretch) model --------------------------------------
        self.I = edges[:, 0].copy()
        self.J = edges[:, 1].copy()
        d0 = vertices[self.J] - vertices[self.I]
        self.L0 = np.linalg.norm(d0, axis=1)
        min_m = np.minimum(self.m[self.I], self.m[self.J])
        self.k_s = 4.0e6 * min_m                     # mass-proportional stiffness
        self.cd_s = 0.30 * np.sqrt(self.k_s * min_m)  # stretch damping

        # ---- bending joints --------------------------------------------
        nbrs = {i: [] for i in range(self.n)}
        edge_lookup = {}
        for e, (i, j) in enumerate(edges):
            nbrs[i].append(j)
            nbrs[j].append(i)
            edge_lookup[(min(i, j), max(i, j))] = e

        E_bend = 1.0e8
        edge_EI = E_bend * np.pi * radius_by_level[np.maximum(levels[self.I], levels[self.J])] ** 4 / 4.0

        A_list, B_list, C_list, kb_list, th0_list, cb_list = [], [], [], [], [], []
        for b in range(self.n):
            ns = nbrs[b]
            if len(ns) < 2:
                continue
            for a, c in combinations(sorted(ns), 2):
                e1 = edge_lookup[(min(a, b), max(a, b))]
                e2 = edge_lookup[(min(b, c), max(b, c))]
                kb = min(edge_EI[e1], edge_EI[e2])
                u0 = vertices[a] - vertices[b]
                w0 = vertices[c] - vertices[b]
                lu = np.linalg.norm(u0)
                lw = np.linalg.norm(w0)
                if lu < 1e-9 or lw < 1e-9:
                    continue
                sin0 = np.linalg.norm(np.cross(u0, w0))
                cos0 = np.clip(np.dot(u0, w0) / (lu * lw), -1.0, 1.0)
                th0 = math.atan2(sin0, cos0)
                m_eff = min(self.m[a], self.m[b], self.m[c])
                l_eff = min(lu, lw)
                cb = 0.25 * math.sqrt(max(kb * m_eff * l_eff * l_eff, 0.0))
                A_list.append(a); B_list.append(b); C_list.append(c)
                kb_list.append(kb); th0_list.append(th0); cb_list.append(cb)

        self.A = np.asarray(A_list, dtype=int)
        self.B = np.asarray(B_list, dtype=int)
        self.C = np.asarray(C_list, dtype=int)
        self.kb = np.asarray(kb_list, dtype=float)
        self.th0 = np.asarray(th0_list, dtype=float)
        self.cb = np.asarray(cb_list, dtype=float)

        deg = np.zeros(self.n)
        for i, j in edges:
            deg[i] += 1.0
            deg[j] += 1.0
        self.deg = np.maximum(deg, 1.0)

        # ---- loads ------------------------------------------------------
        self.g = np.asarray(cfg["gravity"], dtype=float)
        wind = cfg["wind"]
        wd = np.asarray(wind["direction"], dtype=float)
        self.wind_dir = wd / max(np.linalg.norm(wd), 1e-12)
        self.wk_t = np.asarray([k["time"] for k in wind["keyframes"]], dtype=float)
        self.wk_v = np.asarray([k["speed"] for k in wind["keyframes"]], dtype=float)
        self.phase_offsets = np.asarray(wind["branch_phase_offsets"], dtype=float)
        self.gust_f = float(wind["gust_frequency_hz"])
        self.gust_amp = 0.22

        rho_air, cd = 1.225, 1.2
        self.Cw = 0.5 * rho_air * cd * 0.13   # effective bulk drag factor
        self.area = 2.0 * radii * lumped       # branch-area weighted profile
        z_max = max(float(vertices[:, 2].max()), 1e-6)
        self.hweight = 0.35 + 0.65 * vertices[:, 2] / z_max
        self.lin_damp = 0.05

        # ---- integration / budget state ---------------------------------
        self.h = 2.5e-4
        self.imp = np.zeros(3)
        self.angimp = np.zeros(3)
        self.root_pos = vertices[0].copy()

        self.ref_momentum = max(self.m_tot * 5.0, 1e-6)
        self.ref_ang_momentum = max(self.m_tot * z_max * 5.0, 1e-6)

        self.lumped = lumped
        self.radii = radii
        self._initialized = True

    # ------------------------------------------------------------------
    def _wind_speed(self, t: float) -> float:
        return float(np.interp(t, self.wk_t, self.wk_v))

    def _external_forces(self, t: float, v: np.ndarray) -> np.ndarray:
        vw = self._wind_speed(t)
        ph = self.phase_offsets[self.branch_id % self.phase_offsets.size]
        veff = vw * (1.0 + self.gust_amp * np.sin(2.0 * np.pi * self.gust_f * t + ph))
        uw = self.wind_dir[None, :] * veff[:, None]
        urel = uw - v
        s = np.linalg.norm(urel, axis=1)
        fw = self.Cw * self.area * self.hweight * s
        f = self.m[:, None] * self.g[None, :]
        f += fw[:, None] * urel
        f -= self.lin_damp * self.m[:, None] * v
        return f

    def _internal_forces(self) -> np.ndarray:
        x, v = self.x, self.v
        f = np.zeros((self.n, 3))

        # stretch springs
        d = x[self.J] - x[self.I]
        L = np.linalg.norm(d, axis=1)
        Ls = np.maximum(L, 1e-9)
        e = d / Ls[:, None]
        vrel = v[self.J] - v[self.I]
        proj = np.einsum("ij,ij->i", vrel, e)
        fmag = self.k_s * (L - self.L0) + self.cd_s * proj
        fj = -fmag[:, None] * e
        np.add.at(f, self.J, fj)
        np.add.at(f, self.I, -fj)

        # bending joints
        A, B, C = self.A, self.B, self.C
        u = x[A] - x[B]
        w = x[C] - x[B]
        Lu = np.maximum(np.linalg.norm(u, axis=1), 1e-9)
        Lw = np.maximum(np.linalg.norm(w, axis=1), 1e-9)
        uh = u / Lu[:, None]
        wh = w / Lw[:, None]
        nv = np.cross(u, w)
        Ln = np.maximum(np.linalg.norm(nv, axis=1), 1e-9)
        nh = nv / Ln[:, None]
        cosv = np.clip(np.einsum("ij,ij->i", uh, wh), -1.0, 1.0)
        theta = np.arctan2(Ln, cosv)
        om_w = np.einsum("ij,ij->i", v[C] - v[B], np.cross(nh, wh)) / Lw
        om_u = np.einsum("ij,ij->i", v[A] - v[B], np.cross(nh, uh)) / Lu
        thdot = om_w - om_u
        tau = self.kb * (self.th0 - theta) - self.cb * thdot
        fc = (tau / Lw)[:, None] * np.cross(nh, wh)
        fa = -(tau / Lu)[:, None] * np.cross(nh, uh)
        np.add.at(f, C, fc)
        np.add.at(f, A, fa)
        np.add.at(f, B, -(fa + fc))
        return f

    def _node_omega(self) -> np.ndarray:
        d = self.x[self.J] - self.x[self.I]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        e = d / L[:, None]
        we = np.cross(e, self.v[self.J] - self.v[self.I]) / L[:, None]
        om = np.zeros((self.n, 3))
        np.add.at(om, self.I, we)
        np.add.at(om, self.J, we)
        return om / self.deg[:, None]

    def _step(self, h: float) -> None:
        t_new = self.t + h
        f_int = self._internal_forces()
        f_ext = self._external_forces(t_new, self.v)
        acc = (f_int + f_ext) / self.m[:, None]
        self.v += h * acc
        self.x += h * self.v

        # pinned root (fixed_root_vertex)
        self.v[0] = 0.0
        self.x[0] = self.root_pos

        # safety clamp (keeps states finite under extreme gusts)
        vmag = np.linalg.norm(self.v, axis=1)
        scale = np.minimum(1.0, 60.0 / np.maximum(vmag, 1e-12))
        self.v *= scale[:, None]
        self.x[0] = self.root_pos
        self.v[0] = 0.0

        # budget accumulation: gravity+wind (distributed) and root reaction
        reaction = -(f_int[0] + f_ext[0])
        ftot = f_ext.sum(axis=0) + reaction
        self.imp += ftot * h
        ltot = np.cross(self.x, f_ext).sum(axis=0) + np.cross(self.x[0], reaction)
        self.angimp += ltot * h

        # orientation quaternions from nodal angular velocity
        om = self._node_omega()
        wq = np.concatenate([om, np.zeros((self.n, 1))], axis=1)
        qd = 0.5 * _qmul(wq, self.q)
        self.q = self.q + h * qd
        nq = np.linalg.norm(self.q, axis=1)
        self.q /= np.maximum(nq, 1e-12)[:, None]

        self.t = t_new
        self.nsteps += 1

    # ------------------------------------------------------------------
    def _inertia_tensor(self) -> np.ndarray:
        d = self.x[self.J] - self.x[self.I]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        e = d / L[:, None]
        tang = np.zeros((self.n, 3))
        np.add.at(tang, self.I, e)
        np.add.at(tang, self.J, e)
        tn = np.linalg.norm(tang, axis=1)
        fallback = tn < 1e-9
        tang[fallback] = np.array([0.0, 0.0, 1.0])
        tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-12)[:, None]

        ref = np.where(
            np.abs(tang[:, 2:3]) < 0.9,
            np.broadcast_to(np.array([0.0, 0.0, 1.0]), (self.n, 3)),
            np.broadcast_to(np.array([1.0, 0.0, 0.0]), (self.n, 3)),
        )
        e1 = np.cross(ref, tang)
        e1 /= np.maximum(np.linalg.norm(e1, axis=1), 1e-12)[:, None]
        e2 = np.cross(tang, e1)

        a2 = (self.lumped / math.sqrt(12.0)) ** 2
        c2 = (self.radii / math.sqrt(2.0)) ** 2
        inertia = np.zeros((self.n, 3, 3))
        for vec, coef in ((e1, a2), (e2, a2), (tang, c2)):
            inertia += (
                self.m[:, None, None]
                * coef[:, None, None]
                * vec[:, :, None]
                * vec[:, None, :]
            )
        return inertia

    def _frame(self, t: float) -> StateFrame:
        f_ext = self._external_forces(t, self.v)
        f_int = self._internal_forces()

        # term 0: distributed gravity + wind + damping
        f0 = f_ext.sum(axis=0)
        wsum = float(np.abs(f_ext).sum())
        if wsum > 1e-9:
            pos0 = (np.abs(f_ext).sum(axis=1)[:, None] * self.x).sum(axis=0) / wsum
        else:
            pos0 = (self.m[:, None] * self.x).sum(axis=0) / self.m.sum()
        couple0 = np.cross(self.x, f_ext).sum(axis=0) - np.cross(pos0, f0)

        # term 1: root support reaction (point force)
        reaction = -(f_int[0] + f_ext[0])

        ext_force = np.stack([f0, reaction])[None, :, :]
        ext_couple = np.stack([couple0, np.zeros(3)])[None, :, :]
        force_pos = np.stack([pos0, self.x[0]])[None, :, :]

        attributes = {k: np.asarray(v).copy() for k, v in self.base_attributes.items()}
        attributes["inertia_tensor"] = self._inertia_tensor()
        attributes["mass_quantity"] = self.m.copy()

        fields = {
            "budget_cumulative_external_angular_impulse": self.angimp[None, :].copy(),
            "budget_cumulative_external_impulse": self.imp[None, :].copy(),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_couple": ext_couple,
            "budget_external_force": ext_force,
            "budget_force_position": force_pos,
            "cumulative_mass_source": np.zeros(2),
        }
        scalars = {
            "wind_speed": self._wind_speed(t),
            "gravity_magnitude": float(np.linalg.norm(self.g)),
            "reference_momentum": self.ref_momentum,
            "reference_angular_momentum": self.ref_ang_momentum,
        }

        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self._node_omega(),
            mass=self.m.copy(),
            body_id=self.branch_id.astype(np.int64).copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("solver not initialized")
        t_target = max(float(time), self.t)
        dt = t_target - self.t
        if dt > 0.0:
            n = max(1, int(math.ceil(dt / self.h - 1e-9)))
            h = dt / n
            for _ in range(n):
                self._step(h)
            self.t = t_target
        return self._frame(self.t)

    def diagnostics(self) -> dict:
        return {
            "solver": "nodal discrete elastic rod network",
            "time": self.t,
            "substeps": self.nsteps,
            "nodes": self.n,
            "edges": int(len(self.I)),
            "bending_joints": int(len(self.A)),
            "total_mass": self.m_tot,
            "seed": self.seed,
        }
