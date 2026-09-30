"""Self-contained curtain/ball cloth solver (no external physics engine).

Model
-----
* Thin sheet discretised as a mass lattice with structural, shear and flexion
  (bending) springs, viscous spring damping and linear air drag.
* Explicit symplectic-Euler substepping (dt = sample_dt / substeps) with
  Jacobi strain limiting so the sheet behaves near-inextensibly.
* Contacts: frozen kinematic sphere, ground plane and cloth self-contact,
  each resolved with non-penetration projection plus Coulomb friction.
* Initial condition: the pinned curtain is wrapped around the ball (slanted
  span, concentric wrap arc, rolled hem) so gravity drives the flip and
  unravel; no keyframes or scripted motion are used.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _accumulate(out, idx_a, idx_b, vec, n):
    """out[idx_a] += vec ; out[idx_b] -= vec (duplicate-index safe)."""
    for k in range(3):
        w = np.ascontiguousarray(vec[:, k], dtype=float)
        out[:, k] += np.bincount(idx_a, weights=w, minlength=n)
        out[:, k] -= np.bincount(idx_b, weights=w, minlength=n)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        cloth = dict(cfg.get("cloth", {}) or {})
        grid = [int(q) for q in cloth.get("grid_shape", [17, 17])]
        self.nx, self.ny = grid[0], grid[1]
        size = [float(q) for q in cloth.get("size", [3.0, 2.5])]
        self.sx, self.sz = size[0], size[1]
        self.center = np.asarray(cloth.get("center", [0.0, 0.0, 1.55]), dtype=float)
        self.thickness = float(cloth.get("thickness", 0.015))
        self.total_mass = float(cloth.get("total_mass", 0.32))
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.substeps = 160
        self.dt = self.sample_dt / self.substeps
        self.seed = int(seed)
        self._read_geometry(cfg)
        self._build_lattice()
        self._build_params()
        self._build_springs()
        self._build_pairs()
        self._build_stencils()
        self._build_initial_state()
        self.step_count = 0
        self.time = 0.0
        self._max_speed = 0.0
        self._min_gap = 0.0
    def _read_geometry(self, cfg: dict) -> None:
        """Pick up the frozen sphere / ground colliders."""
        self.sphere_center = np.array([0.0, 0.0, 0.68])
        self.sphere_radius = 0.62
        self.ground_top = 0.0
        geo = cfg.get("scene_geometry") or {}
        for obj in (geo.get("objects") or []):
            if not isinstance(obj, dict):
                continue
            g = obj.get("geometry") or {}
            tr = obj.get("transform") or {}
            t = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            gtype = str(g.get("type", ""))
            if gtype == "sphere" and obj.get("collision_enabled", True):
                self.sphere_center = t.copy()
                self.sphere_radius = float(g.get("radius", 0.62))
            elif gtype == "box" and obj.get("collision_enabled", True):
                he = np.asarray(g.get("half_extents", [1.0, 1.0, 0.05]), dtype=float)
                self.ground_top = float(t[2] + he[2])

    def _build_lattice(self) -> None:
        nx, ny = self.nx, self.ny
        xs = np.linspace(-0.5 * self.sx, 0.5 * self.sx, nx)
        zs = np.linspace(-0.5 * self.sz, 0.5 * self.sz, ny)
        xx, zz = np.meshgrid(xs, zs, indexing="xy")
        self.n = nx * ny
        # Material (rest) coordinates: u along x, v along the hanging direction.
        self.mat_u = xx.ravel() + self.center[0]
        self.mat_v = zz.ravel() + self.center[2]
        self.du = float(xs[1] - xs[0]) if nx > 1 else 1.0
        self.dv = float(zs[1] - zs[0]) if ny > 1 else 1.0
        self.col = np.tile(np.arange(nx), ny)          # i index (x)
        self.row = np.repeat(np.arange(ny), nx)        # j index (hanging)
        # Arc length measured from the pinned top edge, in material space.
        self.arc = (ny - 1 - self.row).astype(float) * self.dv
        self.mass = np.full(self.n, self.total_mass / self.n)
        self.inv_mass = 1.0 / self.mass
        self.pinned = self.row == (ny - 1)
        self.free = ~self.pinned
        self.boundary = (
            (self.col == 0) | (self.col == nx - 1) | (self.row == 0) | (self.row == ny - 1)
        ).astype(np.int32)

    def _build_params(self) -> None:
        self.k_struct = 1400.0
        self.k_shear = 220.0
        self.k_bend = 0.9
        self.c_struct = 0.55
        self.c_shear = 0.14
        self.c_bend = 0.02
        self.drag = 0.0012
        self.mu_sphere = 0.35
        self.mu_ground = 0.55
        self.mu_self = 0.25
        self.strain_min = 0.90
        self.strain_max = 1.04
        self.strain_iters = 3
        self.self_dist = 0.085          # enforced layer separation
        self.contact_eps = 0.5 * self.thickness + 0.008
        self.max_speed = 22.0
        self.max_disp = 0.02

    def _idx(self, i, j):
        return j * self.nx + i

    def _build_springs(self) -> None:
        nx, ny = self.nx, self.ny
        s_a, s_b, s_k, s_c = [], [], [], []

        def add(ia, ja, ib, jb, k, c):
            s_a.append(self._idx(ia, ja))
            s_b.append(self._idx(ib, jb))
            s_k.append(k)
            s_c.append(c)

        for j in range(ny):
            for i in range(nx):
                if i + 1 < nx:
                    add(i, j, i + 1, j, self.k_struct, self.c_struct)
                if j + 1 < ny:
                    add(i, j, i, j + 1, self.k_struct, self.c_struct)
        self.n_struct = len(s_a)
        for j in range(ny):
            for i in range(nx):
                if i + 1 < nx and j + 1 < ny:
                    add(i, j, i + 1, j + 1, self.k_shear, self.c_shear)
                    add(i + 1, j, i, j + 1, self.k_shear, self.c_shear)
        for j in range(ny):
            for i in range(nx):
                if i + 2 < nx:
                    add(i, j, i + 2, j, self.k_bend, self.c_bend)
                if j + 2 < ny:
                    add(i, j, i, j + 2, self.k_bend, self.c_bend)
        self.sp_a = np.asarray(s_a, dtype=np.int64)
        self.sp_b = np.asarray(s_b, dtype=np.int64)
        self.sp_k = np.asarray(s_k, dtype=float)
        self.sp_c = np.asarray(s_c, dtype=float)
        # Rest lengths from the undeformed material lattice.
        pu = np.column_stack((self.mat_u, self.mat_v))
        self.sp_l0 = np.linalg.norm(pu[self.sp_a] - pu[self.sp_b], axis=1)
        self.st_a = self.sp_a[: self.n_struct]
        self.st_b = self.sp_b[: self.n_struct]
        self.st_l0 = self.sp_l0[: self.n_struct]

    def _build_pairs(self) -> None:
        """Vertex pairs eligible for self-contact (skip lattice neighbours)."""
        ia, ib = np.triu_indices(self.n, k=1)
        far = (np.abs(self.col[ia] - self.col[ib]) > 1) | (np.abs(self.row[ia] - self.row[ib]) > 1)
        self.pair_a = ia[far].astype(np.int64)
        self.pair_b = ib[far].astype(np.int64)

    def _build_stencils(self) -> None:
        """Central/one-sided finite-difference neighbours for the surface gradient."""
        nx, ny = self.nx, self.ny
        i, j = self.col, self.row
        ip = np.minimum(i + 1, nx - 1)
        im = np.maximum(i - 1, 0)
        jp = np.minimum(j + 1, ny - 1)
        jm = np.maximum(j - 1, 0)
        self.iu_p = (j * nx + ip).astype(np.int64)
        self.iu_m = (j * nx + im).astype(np.int64)
        self.iv_p = (jp * nx + i).astype(np.int64)
        self.iv_m = (jm * nx + i).astype(np.int64)
        self.hu = (self.mat_u[self.iu_p] - self.mat_u[self.iu_m])
        self.hv = (self.mat_v[self.iv_p] - self.mat_v[self.iv_m])
        self.hu[self.hu == 0.0] = self.du
        self.hv[self.hv == 0.0] = self.dv
    # -------------------------------------------------------- initial wrap
    def _wrap_path(self, s):
        """Analytic (y, z) curve: slanted span -> concentric wrap -> rolled hem."""
        cy, cz = float(self.sphere_center[1]), float(self.sphere_center[2])
        rw = self.sphere_radius + 0.02
        z_top = self.center[2] + 0.5 * self.sz
        theta0 = np.deg2rad(20.0)
        p0 = np.array([cy - rw * np.sin(theta0), cz + rw * np.cos(theta0)])
        pin = np.array([0.0, z_top])
        span = p0 - pin
        s_a = float(np.linalg.norm(span))
        d_a = span / max(s_a, 1e-12)
        rf = 0.09
        l_fold = np.pi * rf
        rem = max(self.sz - s_a, 0.0)
        lw = min(max(rem - l_fold - 0.05, 0.0), (np.deg2rad(100.0) - theta0) * rw)
        theta_e = theta0 + lw / rw
        psi_e = 0.5 * np.pi - theta_e
        p_e = np.array([cy - rw * np.sin(theta_e), cz + rw * np.cos(theta_e)])
        normal = np.array([-np.sin(theta_e), np.cos(theta_e)])
        fold_c = p_e + rf * normal
        psi_f = psi_e + np.pi
        p_f = fold_c + rf * np.array([np.cos(psi_f), -np.sin(psi_f)])
        d_f = np.array([-np.sin(psi_f), -np.cos(psi_f)])
        s_b = s_a + lw
        s_c = s_b + l_fold

        s = np.asarray(s, dtype=float)
        out = np.empty((s.size, 2), dtype=float)
        m_a = s <= s_a
        m_b = (s > s_a) & (s <= s_b)
        m_c = (s > s_b) & (s <= s_c)
        m_d = s > s_c
        out[m_a] = pin + s[m_a, None] * d_a
        th = theta0 + (s[m_b] - s_a) / rw
        out[m_b] = np.column_stack((cy - rw * np.sin(th), cz + rw * np.cos(th)))
        psi = psi_e + (s[m_c] - s_b) / rf
        out[m_c] = fold_c + rf * np.column_stack((np.cos(psi), -np.sin(psi)))
        out[m_d] = p_f + (s[m_d] - s_c)[:, None] * d_f
        return out[:, 0], out[:, 1]

    def _build_initial_state(self) -> None:
        rng = np.random.default_rng(self.seed)
        s = self.arc
        y, z = self._wrap_path(s)
        pos = np.column_stack((self.mat_u.copy(), y, z))
        # Deterministic symmetry-breaking so folds and wrinkles develop.
        ramp = np.clip(s / max(self.sz, 1e-9), 0.0, 1.0)
        pos[:, 1] += 0.016 * np.sin(2.0 * np.pi * 2.0 * self.mat_u / self.sx) * ramp
        pos[:, 0] += 0.006 * np.sin(2.0 * np.pi * 1.5 * s / self.sz) * ramp
        pos += rng.normal(0.0, 0.0015, size=pos.shape) * ramp[:, None]
        # Exact pins along the top edge.
        z_top = self.center[2] + 0.5 * self.sz
        pos[self.pinned, 0] = self.mat_u[self.pinned]
        pos[self.pinned, 1] = 0.0
        pos[self.pinned, 2] = z_top
        # Remove any residual sphere / ground penetration.
        rel = pos - self.sphere_center[None, :]
        dist = np.linalg.norm(rel, axis=1)
        rmin = self.sphere_radius + self.contact_eps
        bad = (dist < rmin) & self.free
        if np.any(bad):
            nrm = rel[bad] / np.maximum(dist[bad, None], 1e-12)
            pos[bad] = self.sphere_center[None, :] + nrm * rmin
        floor = self.ground_top + self.contact_eps
        pos[:, 2] = np.where(self.free & (pos[:, 2] < floor), floor, pos[:, 2])
        self.pos = np.ascontiguousarray(pos, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.pin_pos = self.pos[self.pinned].copy()
        self.ext_force = self.mass[:, None] * self.gravity[None, :]
    # --------------------------------------------------------------- physics
    def _spring_forces(self, force) -> None:
        a, b = self.sp_a, self.sp_b
        d = self.pos[a] - self.pos[b]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        length = np.maximum(length, 1e-9)
        unit = d / length[:, None]
        dv = self.vel[a] - self.vel[b]
        scal = self.sp_k * (length - self.sp_l0) + self.sp_c * np.einsum("ij,ij->i", dv, unit)
        _accumulate(force, a, b, -scal[:, None] * unit, self.n)

    def _resolve(self, idx, nrm, depth, mu) -> None:
        """Non-penetration projection + Coulomb friction for unique indices."""
        self.pos[idx] += nrm * depth[:, None]
        v = self.vel[idx]
        vn = np.einsum("ij,ij->i", v, nrm)
        vt = v - vn[:, None] * nrm
        vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        scale = np.clip(
            1.0 - mu * np.abs(np.minimum(vn, 0.0)) / np.maximum(vt_mag, 1e-9), 0.0, 1.0
        )
        self.vel[idx] = vt * scale[:, None] + np.maximum(vn, 0.0)[:, None] * nrm

    def _obstacle_contacts(self) -> None:
        rel = self.pos - self.sphere_center[None, :]
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        rmin = self.sphere_radius + self.contact_eps
        gap = dist - rmin
        idx = np.flatnonzero((gap < 0.0) & self.free)
        if idx.size:
            nrm = rel[idx] / np.maximum(dist[idx, None], 1e-12)
            self._resolve(idx, nrm, -gap[idx], self.mu_sphere)
        floor = self.ground_top + self.contact_eps
        gz = self.pos[:, 2] - floor
        idx = np.flatnonzero((gz < 0.0) & self.free)
        if idx.size:
            nrm = np.zeros((idx.size, 3))
            nrm[:, 2] = 1.0
            self._resolve(idx, nrm, -gz[idx], self.mu_ground)

    def _self_contact(self) -> None:
        pa, pb = self.pair_a, self.pair_b
        d = self.pos[pa] - self.pos[pb]
        l2 = np.einsum("ij,ij->i", d, d)
        act = l2 < self.self_dist * self.self_dist
        if not np.any(act):
            return
        pa, pb, d = pa[act], pb[act], d[act]
        length = np.sqrt(np.maximum(l2[act], 1e-18))
        nrm = d / length[:, None]
        depth = 0.5 * (self.self_dist - length)
        n = self.n
        disp = np.zeros((n, 3))
        for k in range(3):
            w = np.ascontiguousarray(nrm[:, k] * depth)
            disp[:, k] += np.bincount(pa, weights=w, minlength=n)
            disp[:, k] -= np.bincount(pb, weights=w, minlength=n)
        cnt = np.bincount(pa, minlength=n) + np.bincount(pb, minlength=n)
        disp /= np.maximum(cnt, 1.0)[:, None]
        mag = np.sqrt(np.einsum("ij,ij->i", disp, disp))
        over = mag > self.max_disp
        if np.any(over):
            disp[over] *= (self.max_disp / mag[over])[:, None]
        disp[self.pinned] = 0.0
        self.pos += disp
        dv = self.vel[pa] - self.vel[pb]
        vn = np.einsum("ij,ij->i", dv, nrm)
        appr = vn < 0.0
        if not np.any(appr):
            return
        pa, pb, nrm, vn = pa[appr], pb[appr], nrm[appr], vn[appr]
        dvt = dv[appr] - vn[:, None] * nrm
        dvt_mag = np.sqrt(np.einsum("ij,ij->i", dvt, dvt))
        fs = np.clip(self.mu_self * np.abs(vn) / np.maximum(dvt_mag, 1e-9), 0.0, 1.0)
        imp = -0.5 * (vn[:, None] * nrm + fs[:, None] * dvt)
        vdisp = np.zeros((n, 3))
        for k in range(3):
            w = np.ascontiguousarray(imp[:, k])
            vdisp[:, k] += np.bincount(pa, weights=w, minlength=n)
            vdisp[:, k] -= np.bincount(pb, weights=w, minlength=n)
        cnt = np.bincount(pa, minlength=n) + np.bincount(pb, minlength=n)
        vdisp /= np.maximum(cnt, 1.0)[:, None]
        self.vel += vdisp
        self.vel[self.pinned] = 0.0

    def _strain_limit(self) -> None:
        a, b, l0 = self.st_a, self.st_b, self.st_l0
        n = self.n
        cnt = np.maximum(
            np.bincount(a, minlength=n) + np.bincount(b, minlength=n), 1.0
        )[:, None]
        for _ in range(self.strain_iters):
            d = self.pos[a] - self.pos[b]
            length = np.maximum(np.sqrt(np.einsum("ij,ij->i", d, d)), 1e-9)
            target = np.clip(length, self.strain_min * l0, self.strain_max * l0)
            w = (length - target) / length
            act = np.abs(w) > 1e-10
            if not np.any(act):
                break
            aa, bb = a[act], b[act]
            delta = w[act, None] * d[act]
            disp = np.zeros((n, 3))
            for k in range(3):
                ww = np.ascontiguousarray(delta[:, k])
                disp[:, k] -= np.bincount(aa, weights=ww, minlength=n)
                disp[:, k] += np.bincount(bb, weights=ww, minlength=n)
            disp /= cnt
            disp[self.pinned] = 0.0
            self.pos += disp

    def _step(self) -> None:
        dt = self.dt
        force = self.mass[:, None] * self.gravity[None, :] - self.drag * self.vel
        self._spring_forces(force)
        self.vel += dt * (force * self.inv_mass[:, None])
        speed = np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel))
        fast = speed > self.max_speed
        if np.any(fast):
            self.vel[fast] *= (self.max_speed / speed[fast])[:, None]
        self.vel[self.pinned] = 0.0
        self.pos += dt * self.vel
        self._strain_limit()
        self._obstacle_contacts()
        self._self_contact()
        self.pos[self.pinned] = self.pin_pos
        self.vel[self.pinned] = 0.0
        self.step_count += 1
    # --------------------------------------------------------------- outputs
    def _deformation_gradient(self) -> np.ndarray:
        a = (self.pos[self.iu_p] - self.pos[self.iu_m]) / self.hu[:, None]
        b = (self.pos[self.iv_p] - self.pos[self.iv_m]) / self.hv[:, None]
        la = np.maximum(np.sqrt(np.einsum("ij,ij->i", a, a)), 1e-12)
        e1 = a / la[:, None]
        b1 = np.einsum("ij,ij->i", b, e1)
        bperp = b - b1[:, None] * e1
        lb = np.sqrt(np.einsum("ij,ij->i", bperp, bperp))
        grad = np.zeros((self.n, 2, 2), dtype=float)
        grad[:, 0, 0] = la
        grad[:, 0, 1] = b1
        grad[:, 1, 1] = lb
        return grad

    def _contact_signed_distance(self) -> np.ndarray:
        rel = self.pos - self.sphere_center[None, :]
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        gap = dist - (self.sphere_radius + 0.5 * self.thickness)
        floor = self.ground_top + 0.5 * self.thickness
        gap = np.minimum(gap, self.pos[:, 2] - floor)
        pa, pb = self.pair_a, self.pair_b
        d = self.pos[pa] - self.pos[pb]
        pair_gap = np.sqrt(np.einsum("ij,ij->i", d, d)) - self.thickness
        self_gap = np.full(self.n, np.inf)
        np.minimum.at(self_gap, pa, pair_gap)
        np.minimum.at(self_gap, pb, pair_gap)
        gap = np.minimum(gap, np.where(np.isfinite(self_gap), self_gap, gap))
        return np.nan_to_num(gap, nan=0.0, posinf=1e3, neginf=-1e3)

    def _frame(self) -> StateFrame:
        pos = np.ascontiguousarray(self.pos, dtype=float)
        vel = np.ascontiguousarray(self.vel, dtype=float)
        ext = self.mass[:, None] * self.gravity[None, :] - self.drag * vel
        gap = self._contact_signed_distance()
        speed = float(np.max(np.sqrt(np.einsum("ij,ij->i", vel, vel)))) if self.n else 0.0
        self._max_speed = max(self._max_speed, speed)
        self._min_gap = min(self._min_gap, float(np.min(gap)))
        frame = StateFrame(
            float(self.time),
            pos.copy(),
            vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": gap,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": np.ascontiguousarray(ext, dtype=float),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # --------------------------------------------------------------- stepping
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        while self.step_count < target:
            self._step()
        self.time = float(time)
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "self_mass_spring_cloth",
            "nodes": int(self.n),
            "springs": int(self.sp_a.size),
            "self_pairs": int(self.pair_a.size),
            "substep_dt": float(self.dt),
            "steps": int(self.step_count),
            "sim_time": float(self.time),
            "max_speed": float(self._max_speed),
            "min_contact_gap": float(self._min_gap),
        }
