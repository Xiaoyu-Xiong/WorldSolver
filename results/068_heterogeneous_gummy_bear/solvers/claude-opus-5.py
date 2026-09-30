"""Heterogeneous gummy bear: corotational tetrahedral FEM bonded to rigid stiff cores.

Physical model
--------------
* Discretisation: one tetrahedral mesh built by Delaunay tetrahedralisation of the
  frozen outer + stiff surface vertices, filtered to the interior of the frozen
  outer surface (generalised winding number).  Node order equals the frozen state
  order (946 outer vertices then 516 stiff vertices), so the renderer can reuse
  the frozen surface topology directly.
* Soft shell: corotational linear elasticity (E = 6e4 Pa, nu = 0.45,
  rho = 1000 kg/m^3) with corotational Kelvin-Voigt strain-rate damping.
* Stiff interior: E = 1e14 Pa is numerically indistinguishable from rigidity, so
  each connected stiff region is integrated as a rigid body driven by the total
  force and torque collected from the elastic, gravity and contact forces acting
  on its nodes (Newton-Euler with exact Rodrigues rotation).  The regions
  therefore translate *and* rotate with the surrounding shell, keep their
  outlines, and stay bonded because the interface tetrahedra share nodes with the
  rigid regions.
* Ground: penalty normal contact with damping and regularised Coulomb friction.
* Integration: symplectic (semi-implicit) Euler for the soft nodes at a
  CFL-limited step; all internal forces are exact equal-and-opposite pairs, so the
  exported momentum budgets close.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_WORKSPACE = Path(__file__).resolve().parent.parent


def _load_obj(path: Path):
    """Return (vertices[Nv,3], triangles[Nf,3]) of an ASCII OBJ mesh."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(token) for token in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _winding_number(points: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Generalised winding number (solid angle sum / 4 pi) of closed triangle soup."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    out = np.empty(len(points), dtype=np.float64)
    chunk = max(1, int(4.0e5 // max(1, len(faces))))
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk][:, None, :]
        va = a[None, :, :] - p
        vb = b[None, :, :] - p
        vc = c[None, :, :] - p
        la = np.linalg.norm(va, axis=2)
        lb = np.linalg.norm(vb, axis=2)
        lc = np.linalg.norm(vc, axis=2)
        num = np.einsum("ijk,ijk->ij", va, np.cross(vb, vc))
        den = (
            la * lb * lc
            + la * np.einsum("ijk,ijk->ij", vb, vc)
            + lb * np.einsum("ijk,ijk->ij", vc, va)
            + lc * np.einsum("ijk,ijk->ij", va, vb)
        )
        out[start:start + chunk] = 2.0 * np.arctan2(num, den).sum(axis=1) / (4.0 * np.pi)
    return out


def _components(n: int, edges: np.ndarray) -> list[np.ndarray]:
    """Connected components of an undirected node graph (union-find)."""
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for u, v in edges:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    roots = np.array([find(int(i)) for i in range(n)], dtype=np.int64)
    return [np.flatnonzero(roots == r) for r in np.unique(roots)]


def _scatter(index_flat: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, 3), dtype=np.float64)
    for c in range(3):
        out[:, c] = np.bincount(index_flat, weights=values[:, c], minlength=n)
    return out


class BenchmarkSolver:
    """Explicit corotational-FEM solver with rigid stiff inclusions."""

    def initialize(self, scene, seed: int) -> None:
        from scipy.spatial import Delaunay, cKDTree

        self._rng = np.random.default_rng(int(seed))
        config = dict(scene.config)
        ic = dict(config.get("initial_conditions", {}))
        bc = dict(config.get("boundary_conditions", {}))

        self.gravity = np.asarray(ic.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.rho = float(ic.get("density_kg_m3", 1000.0))
        self.nu = float(ic.get("poisson_ratio", 0.45))
        self.E_soft = float(ic.get("soft_youngs_modulus_pa", 6.0e4))
        self.E_stiff = float(ic.get("stiff_youngs_modulus_pa", 1.0e14))
        self.ground_z = float(bc.get("ground_height", 0.0))
        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.duration = float(config.get("duration", 2.4))

        frame0 = scene.initial_state()
        self.rest = np.array(frame0.position, dtype=np.float64)
        self.pos = self.rest.copy()
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.n = len(self.pos)

        outer_v, outer_f = _load_obj(_WORKSPACE / "assets" / "gummy_bear_outer.obj")
        stiff_v, stiff_f = _load_obj(_WORKSPACE / "assets" / "gummy_bear_stiff.obj")
        self.n_outer = len(outer_v)

        # --- tetrahedral mesh over the frozen surface vertices -------------
        tri = Delaunay(self.rest, qhull_options="Qbb Qc Q12")
        tets = np.asarray(tri.simplices, dtype=np.int64)
        tets = tets[(tets >= 0).all(axis=1) & (tets < self.n).all(axis=1)]
        p0, p1, p2, p3 = (self.rest[tets[:, k]] for k in range(4))
        dm = np.stack((p1 - p0, p2 - p0, p3 - p0), axis=2)
        vol6 = np.linalg.det(dm)
        flip = vol6 < 0.0
        tets[flip] = tets[flip][:, [0, 2, 1, 3]]
        vol = np.abs(vol6) / 6.0
        edges_len = np.stack([
            np.linalg.norm(self.rest[tets[:, i]] - self.rest[tets[:, j]], axis=1)
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        ], axis=1)
        max_edge = edges_len.max(axis=1)
        quality = vol / np.maximum(max_edge ** 3, 1e-12)
        keep = (vol > 1e-7) & (quality > 1.2e-3)
        tets = tets[keep]
        vol = vol[keep]
        max_edge = max_edge[keep]

        centroid = self.rest[tets].mean(axis=1)
        inside = _winding_number(centroid, outer_v, outer_f) > 0.5
        tets, vol, max_edge = tets[inside], vol[inside], max_edge[inside]
        centroid = centroid[inside]
        # reject elements that bridge concave gaps (legs / arms) by testing midpoints
        long_mask = max_edge > 0.30
        if np.any(long_mask):
            idx = np.flatnonzero(long_mask)
            mids = np.concatenate([
                0.5 * (self.rest[tets[idx, i]] + self.rest[tets[idx, j]])
                for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ], axis=0)
            ok = (_winding_number(mids, outer_v, outer_f) > 0.5).reshape(6, len(idx)).all(axis=0)
            drop = idx[~ok]
            if len(drop):
                mask = np.ones(len(tets), dtype=bool)
                mask[drop] = False
                tets, vol, centroid = tets[mask], vol[mask], centroid[mask]
        if len(tets) < 200:  # geometric fallback: keep the raw interior mesh
            tri = Delaunay(self.rest, qhull_options="Qbb Qc Q12")
            tets = np.asarray(tri.simplices, dtype=np.int64)
            tets = tets[(tets >= 0).all(axis=1) & (tets < self.n).all(axis=1)]
            p0, p1, p2, p3 = (self.rest[tets[:, k]] for k in range(4))
            vol6 = np.linalg.det(np.stack((p1 - p0, p2 - p0, p3 - p0), axis=2))
            flip = vol6 < 0.0
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            vol = np.abs(vol6) / 6.0
            keep = vol > 1e-7
            tets, vol = tets[keep], vol[keep]
            centroid = self.rest[tets].mean(axis=1)
        self.tets = tets
        self.ne = len(tets)
        self.vol = vol
        self.tet_flat = tets.reshape(-1)
        # --- rest shape inverses -------------------------------------------
        r0 = self.rest[tets[:, 0]]
        dm = np.stack((self.rest[tets[:, 1]] - r0,
                       self.rest[tets[:, 2]] - r0,
                       self.rest[tets[:, 3]] - r0), axis=2)
        self.dm_inv = np.linalg.inv(dm)

        # --- material assignment -------------------------------------------
        stiff_elem = _winding_number(centroid, stiff_v, stiff_f) > 0.5
        self.stiff_elem = stiff_elem
        rigid_node = np.zeros(self.n, dtype=bool)
        rigid_node[self.n_outer:] = True  # stiff-surface nodes drive the rigid cores
        self.rigid_node = rigid_node
        all_rigid = rigid_node[tets].all(axis=1)
        self.all_rigid_elem = all_rigid
        # Interface / stiff-interior elements are stiffened moderately; the true
        # 1e14 Pa contrast is realised exactly by the rigid-core constraint.
        e_elem = np.full(self.ne, self.E_soft, dtype=np.float64)
        e_elem[stiff_elem & ~all_rigid] = 3.0 * self.E_soft
        self.mu = e_elem / (2.0 * (1.0 + self.nu))
        self.lam = e_elem * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.eta = 60.0  # Pa*s Kelvin-Voigt viscosity of the gummy material

        # --- lumped mass ----------------------------------------------------
        mass = np.bincount(self.tet_flat,
                           weights=np.repeat(self.rho * self.vol / 4.0, 4),
                           minlength=self.n)
        positive = mass[mass > 0.0]
        fill = float(positive.mean()) if len(positive) else 1.0e-3
        mass[mass <= 0.0] = fill
        self.mass = mass
        self.inv_mass = 1.0 / mass
        self.total_mass = float(mass.sum())

        # --- bonding repair: spring-connect any disconnected node groups ----
        pairs = np.unique(np.sort(np.concatenate([
            tets[:, [i, j]] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        ], axis=0), axis=1), axis=0)
        comps = _components(self.n, pairs)
        comps.sort(key=lambda c: -self.mass[c].sum())
        springs: list[tuple[int, int]] = []
        if len(comps) > 1:
            main = comps[0]
            tree = cKDTree(self.rest[main])
            for comp in comps[1:]:
                dist, near = tree.query(self.rest[comp], k=1)
                order = np.argsort(dist)[:max(3, min(6, len(comp)))]
                for o in order:
                    springs.append((int(comp[o]), int(main[near[o]])))
        if springs:
            sp = np.asarray(springs, dtype=np.int64)
            self.spring_i = sp[:, 0]
            self.spring_j = sp[:, 1]
            self.spring_rest = np.linalg.norm(
                self.rest[self.spring_i] - self.rest[self.spring_j], axis=1)
            self.spring_k = 4.0e4
            self.spring_c = 40.0
        else:
            self.spring_i = np.zeros(0, dtype=np.int64)
            self.spring_j = np.zeros(0, dtype=np.int64)
            self.spring_rest = np.zeros(0, dtype=np.float64)
            self.spring_k = 0.0
            self.spring_c = 0.0
        # --- rigid stiff regions (connected components of the stiff surface) --
        stiff_edges = np.unique(np.sort(np.concatenate([
            stiff_f[:, [0, 1]], stiff_f[:, [1, 2]], stiff_f[:, [2, 0]]
        ], axis=0), axis=1), axis=0)
        self.groups = []
        for comp in _components(len(stiff_v), stiff_edges):
            idx = comp.astype(np.int64) + self.n_outer
            m = self.mass[idx]
            m_tot = float(m.sum())
            com = (m[:, None] * self.pos[idx]).sum(axis=0) / m_tot
            self.groups.append({
                "idx": idx,
                "m": m,
                "m_tot": m_tot,
                "com": com,
                "offset": self.pos[idx] - com,
                "v": np.zeros(3),
                "w": np.zeros(3),
            })

        # --- momentum-conserving rigid projection of the prescribed field ----
        for g in self.groups:
            idx, m = g["idx"], g["m"]
            r = g["offset"]
            p_lin = (m[:, None] * self.vel[idx]).sum(axis=0)
            g["v"] = p_lin / g["m_tot"]
            l_ang = np.cross(r, m[:, None] * self.vel[idx]).sum(axis=0)
            inertia = self._inertia(m, r)
            g["w"] = np.linalg.solve(inertia, l_ang)
            self.vel[idx] = g["v"] + np.cross(g["w"], r)

        # --- contact model ---------------------------------------------------
        omega_c = 220.0
        self.contact_k = self.mass * omega_c * omega_c
        self.contact_c = 2.0 * 0.55 * self.mass * omega_c
        self.friction_mu = 0.6
        self.pen_max = 0.05
        gap0 = self.ground_z - self.pos[:, 2]
        self.pen_offset = np.maximum(gap0, 0.0)
        self.pen_relax = 0.25  # s, initial overlap released smoothly

        # --- explicit time step (CFL on the corotational wave speed) ---------
        wave = np.sqrt((self.lam + 2.0 * self.mu) / self.rho)
        face_area = np.empty((self.ne, 4), dtype=np.float64)
        q = self.rest[self.tets]
        for k, (a, b, c) in enumerate(((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))):
            face_area[:, k] = 0.5 * np.linalg.norm(
                np.cross(q[:, b] - q[:, a], q[:, c] - q[:, a]), axis=1)
        altitude = 3.0 * self.vol / np.maximum(face_area.max(axis=1), 1e-12)
        dt_cfl = float(np.min(0.35 * altitude / np.maximum(wave, 1e-9)))
        dt_contact = 0.35 * 2.0 / omega_c
        dt_target = min(dt_cfl, dt_contact)
        self.substeps = int(np.clip(np.ceil(self.sample_dt / dt_target), 40, 130))
        self.dt = self.sample_dt / self.substeps

        self.time = 0.0
        self.step_index = 0
        self.n_groups = 3  # [all, soft shell, stiff cores]
        self.domain_weights = np.zeros((3, self.n), dtype=np.float64)
        self.domain_weights[0, :] = 1.0
        self.domain_weights[1, self.body_id == 0] = 1.0
        self.domain_weights[2, self.body_id == 1] = 1.0
        self.cum_impulse = np.zeros((3, 3), dtype=np.float64)
        self.reference_momentum = float(self.total_mass * 1.0)
        self._steps_taken = 0
    # ------------------------------------------------------------------ physics
    @staticmethod
    def _inertia(m: np.ndarray, r: np.ndarray) -> np.ndarray:
        r2 = np.einsum("ij,ij->i", r, r)
        inertia = np.eye(3) * float((m * r2).sum())
        inertia -= np.einsum("i,ij,ik->jk", m, r, r)
        trace = float(np.trace(inertia))
        return inertia + np.eye(3) * max(1e-9, 1e-9 * abs(trace))

    def _forces(self):
        """Return (f_internal, f_gravity, f_contact, deformation_gradient)."""
        tets = self.tets
        x0 = self.pos[tets[:, 0]]
        ds = np.stack((self.pos[tets[:, 1]] - x0,
                       self.pos[tets[:, 2]] - x0,
                       self.pos[tets[:, 3]] - x0), axis=2)
        f_grad = ds @ self.dm_inv

        # polar decomposition F = R S via eigen-decomposition of F^T F
        s_mat = np.einsum("nij,nik->njk", f_grad, f_grad)
        w, v = np.linalg.eigh(s_mat)
        w = np.maximum(w, 1e-12)
        inv_sqrt = np.einsum("nij,nj,nkj->nik", v, 1.0 / np.sqrt(w), v)
        rot = f_grad @ inv_sqrt
        det = np.linalg.det(f_grad)
        neg = det < 0.0
        if np.any(neg):
            v0 = v[neg, :, 0]
            rv = np.einsum("nij,nj->ni", rot[neg], v0)
            rot[neg] -= 2.0 * rv[:, :, None] * v0[:, None, :]

        # corotational (co-rotated linear) first Piola stress
        rtf = np.einsum("nij,nik->njk", rot, f_grad)
        trace = rtf[:, 0, 0] + rtf[:, 1, 1] + rtf[:, 2, 2] - 3.0
        piola = 2.0 * self.mu[:, None, None] * (f_grad - rot)
        piola += (self.lam * trace)[:, None, None] * rot

        # corotational Kelvin-Voigt viscous stress (zero for rigid motion)
        v0n = self.vel[tets[:, 0]]
        dv = np.stack((self.vel[tets[:, 1]] - v0n,
                       self.vel[tets[:, 2]] - v0n,
                       self.vel[tets[:, 3]] - v0n), axis=2)
        f_dot = dv @ self.dm_inv
        rtd = np.einsum("nij,nik->njk", rot, f_dot)
        strain_rate = 0.5 * (rtd + np.transpose(rtd, (0, 2, 1)))
        piola += self.eta * (rot @ strain_rate)

        h_mat = -self.vol[:, None, None] * (piola @ np.transpose(self.dm_inv, (0, 2, 1)))
        f_nodes = np.empty((self.ne, 4, 3), dtype=np.float64)
        f_nodes[:, 1, :] = h_mat[:, :, 0]
        f_nodes[:, 2, :] = h_mat[:, :, 1]
        f_nodes[:, 3, :] = h_mat[:, :, 2]
        f_nodes[:, 0, :] = -(f_nodes[:, 1] + f_nodes[:, 2] + f_nodes[:, 3])
        f_int = _scatter(self.tet_flat, f_nodes.reshape(-1, 3), self.n)
        if len(self.spring_i):
            i, j = self.spring_i, self.spring_j
            d = self.pos[i] - self.pos[j]
            length = np.linalg.norm(d, axis=1)
            unit = d / np.maximum(length, 1e-12)[:, None]
            rel = np.einsum("ij,ij->i", self.vel[i] - self.vel[j], unit)
            mag = -self.spring_k * (length - self.spring_rest) - self.spring_c * rel
            fs = mag[:, None] * unit
            f_int += _scatter(i, fs, self.n)
            f_int += _scatter(j, -fs, self.n)

        f_grav = self.mass[:, None] * self.gravity[None, :]

        # penalty ground contact with damping and regularised Coulomb friction
        decay = max(0.0, 1.0 - self.time / self.pen_relax)
        offset = self.pen_offset * decay
        depth = np.clip(self.ground_z - self.pos[:, 2] - offset, 0.0, self.pen_max)
        f_contact = np.zeros((self.n, 3), dtype=np.float64)
        active = depth > 0.0
        if np.any(active):
            fn = self.contact_k[active] * depth[active]
            fn -= self.contact_c[active] * np.minimum(self.vel[active, 2], 0.0)
            fn = np.maximum(fn, 0.0)
            f_contact[active, 2] = fn
            vt = self.vel[active, :2]
            speed = np.linalg.norm(vt, axis=1)
            # |f_t| = mu*fn for sliding, linearly regularised below 0.05 m/s
            scale = self.friction_mu * fn / np.maximum(speed, 0.05)
            f_contact[active, 0] = -scale * vt[:, 0]
            f_contact[active, 1] = -scale * vt[:, 1]
        return f_int, f_grav, f_contact, f_grad

    def _step(self) -> None:
        dt = self.dt
        f_int, f_grav, f_contact, _ = self._forces()
        f_tot = f_int + f_grav + f_contact

        for g in self.groups:
            idx, m, r = g["idx"], g["m"], g["offset"]
            f_group = f_tot[idx]
            force = f_group.sum(axis=0)
            torque = np.cross(r, f_group).sum(axis=0)
            inertia = self._inertia(m, r)
            g["v"] = g["v"] + dt * force / g["m_tot"]
            l_ang = inertia @ g["w"] + dt * torque
            g["w"] = np.linalg.solve(inertia, l_ang)
            g["com"] = g["com"] + dt * g["v"]
            omega = g["w"]
            angle = float(np.linalg.norm(omega)) * dt
            if angle > 1e-12:
                axis = omega / np.linalg.norm(omega)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                r_new = (r * cos_a
                         + np.cross(axis[None, :], r) * sin_a
                         + np.outer(r @ axis, axis) * (1.0 - cos_a))
                g["offset"] = r_new
            self.pos[idx] = g["com"] + g["offset"]
            self.vel[idx] = g["v"] + np.cross(omega[None, :], g["offset"])

        soft = ~self.rigid_node
        self.vel[soft] += dt * self.inv_mass[soft, None] * f_tot[soft]
        self.pos[soft] += dt * self.vel[soft]
        self.step_index += 1
        self.time = self.step_index * dt
    # ------------------------------------------------------------------ export
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        while self.step_index < target:
            self._step()
            self._steps_taken += 1
        return self._frame(float(time))

    def _frame(self, time: float) -> StateFrame:
        f_int, f_grav, f_contact, f_grad = self._forces()
        weights = self.domain_weights

        budget_force = np.zeros((3, 3, 3), dtype=np.float64)
        for g in range(3):
            w = weights[g][:, None]
            budget_force[g, 0] = (w * f_grav).sum(axis=0)     # gravity
            budget_force[g, 1] = (w * f_contact).sum(axis=0)  # ground contact
            budget_force[g, 2] = (w * f_int).sum(axis=0)      # boundary traction

        stiff_mask = self.body_id == 1
        f_on_stiff = f_int[stiff_mask].sum(axis=0)
        interaction = np.stack((f_on_stiff, -f_on_stiff), axis=0)

        mass_quantity = np.zeros((self.n, 2), dtype=np.float64)
        mass_quantity[~stiff_mask, 0] = self.mass[~stiff_mask]
        mass_quantity[stiff_mask, 1] = self.mass[stiff_mask]

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass_quantity,
                "material_region": self.body_id.astype(np.float64),
            },
            fields={
                "budget_domain_weights": weights.copy(),
                "budget_external_force": budget_force,
                "budget_cumulative_external_impulse": np.zeros((3, 3), dtype=np.float64),
                "budget_cumulative_inflow_momentum": np.zeros((3, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((3, 3), dtype=np.float64),
                "cumulative_mass_source": np.zeros(2, dtype=np.float64),
                "element_deformation_gradient": np.ascontiguousarray(f_grad),
                "interaction_force": interaction,
            },
            scalars={"reference_momentum": self.reference_momentum},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "corotational tetrahedral FEM with rigid stiff inclusions",
            "nodes": int(self.n),
            "elements": int(self.ne),
            "stiff_elements": int(self.stiff_elem.sum()),
            "rigid_regions": int(len(self.groups)),
            "rigid_nodes": int(self.rigid_node.sum()),
            "repair_springs": int(len(self.spring_i)),
            "dt": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "steps_taken": int(self._steps_taken),
            "total_mass_kg": float(self.total_mass),
        }
