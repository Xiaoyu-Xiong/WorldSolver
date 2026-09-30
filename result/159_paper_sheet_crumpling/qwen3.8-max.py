"""Physically-based thin-sheet (paper) crumpling solver.

Model
-----
A 17x17 vertex triangulated sheet is simulated with a lumped-mass spring
network (structural + shear + two-hop bending springs, axial viscous damping)
integrated by semi-implicit Euler with fixed substeps.  Two kinematic rigid
walls close on the sheet along x, imposing large-scale compression; the sheet
buckles out of plane, develops ridges/vertices and settles into a compact
crumpled mass on the ground.  Collisions (ground box, both wall boxes, and
short-range sheet self-contact) are resolved by robust position projection
with frictional velocity updates after each substep, preventing fold-through.

All required trajectory fields are exported: per-sample mass, velocity,
external force, 2x2 material deformation gradients, signed contact distances,
conserved mass quantities and the frozen contact length scale.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _box_sdf_and_normal(p: np.ndarray, center: np.ndarray, half: np.ndarray):
    """Signed distance (N,3)->(N,) and outward normal (N,3) for an AABB."""
    q = np.abs(p - center) - half
    qpos = np.maximum(q, 0.0)
    outside = np.linalg.norm(qpos, axis=1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), q[:, 2])
    sd = outside + inside
    s = np.sign(p - center)
    s[s == 0] = 1.0
    n = np.zeros_like(p)
    ext = outside > 1e-12
    n[ext] = qpos[ext] * s[ext]
    nn = np.linalg.norm(n[ext], axis=1)
    nn[nn < 1e-12] = 1.0
    n[ext] /= nn[:, None]
    # interior: normal along the axis of the largest q component
    ax = np.argmax(q, axis=1)
    rows = np.arange(len(p))
    n[~ext, :] = 0.0
    n[~ext, rows[~ext], ax[~ext]] = s[~ext, rows[~ext], ax[~ext]]
    return sd, n


class BenchmarkSolver:
    # ---- configuration -------------------------------------------------
    substeps = 20                 # per sample interval
    k_struct = 150.0              # N/m
    k_shear = 60.0
    k_bend = 4.0
    c_spring = 0.4                # axial spring damping
    drag = 0.3                    # global linear air drag (1/s)
    r_contact = 0.0075            # half thickness: boundary push radius
    r_self = 0.045                # self-contact activation radius
    mu_b = 0.35                   # boundary friction (tangential loss)
    mu_s = 0.25                   # self-contact friction
    wall_x0 = 1.62                # initial wall-center offset (inner face 1.50)
    wall_x1 = 0.62                # final wall-center offset (inner face 0.50)
    t_start = 0.25                # compression begins
    t_end = 3.2                   # compression ends

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        nx, ny = int(cloth["grid_shape"][0]), int(cloth["grid_shape"][1])
        sx, sy = float(cloth["size"][0]), float(cloth["size"][1])
        center = np.asarray(cloth["center"], dtype=float)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.thickness = float(cloth.get("thickness", 0.015))
        self.dt = (float(cfg["sample_dt"])) / self.substeps

        # ---- rest configuration (matches scene.initial_state layout) ----
        gx = np.linspace(-sx * 0.5, sx * 0.5, nx)
        gy = np.linspace(-sy * 0.5, sy * 0.5, ny)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        rest = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size))) + center
        self.nx, self.ny = nx, ny
        self.n = rest.shape[0]

        # deterministic symmetry-breaking perturbation (paper is never perfect)
        rng = np.random.default_rng(int(seed))
        self.rest = rest.copy()
        self.x = rest.copy()
        self.x[:, 2] += rng.normal(0.0, 0.012, self.n)
        self.x[:, 0] += rng.normal(0.0, 0.004, self.n)
        self.x[:, 1] += rng.normal(0.0, 0.004, self.n)
        self.v = rng.normal(0.0, 0.02, (self.n, 3))

        m_total = float(cloth["total_mass"])
        self.mass = np.full(self.n, m_total / self.n)
        self.inv_m = 1.0 / self.mass

        boundary = ((np.isclose(xx, gx[0])) | (np.isclose(xx, gx[-1])) |
                    (np.isclose(yy, gy[0])) | (np.isclose(yy, gy[-1]))).ravel().astype(np.int32)
        self.boundary = boundary
        self.body_id = np.zeros(self.n, dtype=np.int32)

        # ---- spring sets on the grid ----
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        i, j = i.ravel(), j.ravel()
        idx = i * ny + j
        sets = []
        # structural
        for di, dj in ((1, 0), (0, 1)):
            ok = (i + di < nx) & (j + dj < ny)
            sets.append((idx[ok], (i[ok] + di) * ny + (j[ok] + dj), self.k_struct))
        # shear diagonals
        for di, dj in ((1, 1), (1, -1)):
            ok = (i + di < nx) & (j + dj >= 0) & (j + dj < ny)
            sets.append((idx[ok], (i[ok] + di) * ny + (j[ok] + dj), self.k_shear))
        # bending (two-hop)
        for di, dj in ((2, 0), (0, 2)):
            ok = (i + di < nx) & (j + dj < ny)
            sets.append((idx[ok], (i[ok] + di) * ny + (j[ok] + dj), self.k_bend))
        a = np.concatenate([s[0] for s in sets])
        b = np.concatenate([s[1] for s in sets])
        k = np.concatenate([np.full(len(s[0]), s[2]) for s in sets])
        self.sp_a, self.sp_b, self.sp_k = a, b, k
        d = self.rest[a] - self.rest[b]
        self.sp_L = np.linalg.norm(d, axis=1)

        # ---- triangles for deformation gradients ----
        ti, tj = [], []
        for I in range(nx - 1):
            for J in range(ny - 1):
                v0 = I * ny + J
                v1 = (I + 1) * ny + J
                v2 = I * ny + (J + 1)
                v3 = (I + 1) * ny + (J + 1)
                ti.append((v0, v1, v2))
                ti.append((v1, v3, v2))
        self.tris = np.asarray(ti, dtype=np.int64)
        # per-triangle material (rest) edge matrix Ar (2x2) and projection basis
        p0 = self.rest[self.tris[:, 0]]
        e1 = self.rest[self.tris[:, 1]] - p0
        e2 = self.rest[self.tris[:, 2]] - p0
        b1 = e1 / np.linalg.norm(e1, axis=1)[:, None]
        b2 = e2 - (np.einsum("ij,ij->i", e2, b1))[:, None] * b1
        b2 /= np.linalg.norm(b2, axis=1)[:, None]
        self.basis = np.stack((b1, b2), axis=1)            # (T,2,3)
        Ar = np.stack((np.einsum("ti,tki->tk", e1, self.basis),
                       np.einsum("ti,tki->tk", e2, self.basis)), axis=1)  # (T,2,2)
        self.Ar_inv = np.linalg.inv(Ar)
        # vertex averaging weights
        self.tri_of_v = [[] for _ in range(self.n)]
        for t, tri in enumerate(self.tris):
            for vidx in tri:
                self.tri_of_v[vidx].append(t)
        self.tri_counts = np.array([max(len(l), 1) for l in self.tri_of_v], dtype=float)

        # ---- static geometry ----
        self.ground_c = np.array([0.0, 0.0, -0.05])
        self.ground_h = np.array([2.4, 2.0, 0.05])
        self.wall_h = np.array([0.12, 1.5, 1.3])
        self.wall_z = np.array([0.0, 0.0, 1.3])

        # precomputed pair index array (upper triangle) for self-contact
        self.ii, self.jj = np.triu_indices(self.n, k=1)

        self.t = 0.0
        self.ext_force = self.mass[:, None] * self.gravity[None, :]

    # ---- kinematic wall schedule --------------------------------------
    def _wall_xc(self, t: float) -> float:
        if t <= self.t_start:
            return self.wall_x0
        if t >= self.t_end:
            return self.wall_x1
        s = (t - self.t_start) / (self.t_end - self.t_start)
        return self.wall_x0 + (self.wall_x1 - self.wall_x0) * (s * s * (3.0 - 2.0 * s))

    def _wall_vc(self, t: float) -> float:
        """d/dt of wall_xc (signed; walls move inward => left wall vx>0)."""
        if t <= self.t_start or t >= self.t_end:
            return 0.0
        s = (t - self.t_start) / (self.t_end - self.t_start)
        ds = 6.0 * s * (1.0 - s) / (self.t_end - self.t_start)
        return (self.wall_x1 - self.wall_x0) * ds  # negative (closing)

    # ---- simulation ----------------------------------------------------
    def _step(self, t: float) -> None:
        dt = self.dt
        x, v = self.x, self.v
        a, b, k, L = self.sp_a, self.sp_b, self.sp_k, self.sp_L

        # spring forces with axial damping
        d = x[a] - x[b]
        dist = np.linalg.norm(d, axis=1)
        dist_safe = np.maximum(dist, 1e-9)
        u = d / dist_safe[:, None]
        rv = v[a] - v[b]
        fmag = k * (dist - L) + self.c_spring * np.einsum("ij,ij->i", rv, u)
        f = fmag[:, None] * u
        fa = np.zeros_like(x)
        np.add.at(fa, a, -f)
        np.add.at(fa, b, f)

        # integrate (semi-implicit Euler + linear drag)
        v += (fa * self.inv_m[:, None] + self.gravity[None, :]) * dt
        v *= np.exp(-self.drag * dt)
        x += v * dt

        # ---- boundary collisions (projection + frictional velocity fix) ----
        xc = self._wall_xc(t)
        vc = self._wall_vc(t)  # negative => closing
        boxes = [
            (self.ground_c.copy(), self.ground_h, np.zeros(3)),
            (np.array([-xc, 0.0, 1.3]), self.wall_h, np.array([-vc, 0.0, 0.0])),
            (np.array([xc, 0.0, 1.3]), self.wall_h, np.array([vc, 0.0, 0.0])),
        ]
        for c, h, vw in boxes:
            sd, nrm = _box_sdf_and_normal(x, c, h)
            pen = sd < self.r_contact
            if not np.any(pen):
                continue
            idx = np.where(pen)[0]
            nn = nrm[idx]
            x[idx] += nn * (self.r_contact - sd[idx])[:, None]
            vrel = v[idx] - vw[None, :]
            vn = np.einsum("ij,ij->i", vrel, nn)
            vt = vrel - vn[:, None] * nn
            # remove inward normal motion (inelastic), Coulomb-ish tangential friction
            scale = np.where(vn < 0.0, 0.0, 1.0)
            v[idx] = vw[None, :] + scale[:, None] * vn[:, None] * nn + (1.0 - self.mu_b) * vt

        # ---- short-range self-contact (keeps layers from interpenetrating) ----
        dvec = x[self.ii] - x[self.jj]
        d2 = np.einsum("ij,ij->i", dvec, dvec)
        hit = np.where(d2 < self.r_self * self.r_self)[0]
        if hit.size:
            dist = np.sqrt(np.maximum(d2[hit], 1e-12))
            u = dvec[hit] / dist[:, None]
            corr = 0.5 * (self.r_self - dist)
            ia, ib = self.ii[hit], self.jj[hit]
            np.add.at(x, ia, u * corr[:, None])
            np.add.at(x, ib, -u * corr[:, None])
            rv = v[ia] - v[ib]
            vn = np.einsum("ij,ij->i", rv, u)
            vt = rv - vn[:, None] * u
            imp = np.where(vn < 0.0, -vn, 0.0)
            dv = 0.5 * (imp[:, None] * u - self.mu_s * vt)
            v[ia] += dv
            v[ib] -= dv

        self.t += dt

    # ---- exports -------------------------------------------------------
    def _deformation_gradients(self) -> np.ndarray:
        p0 = self.x[self.tris[:, 0]]
        e1 = self.x[self.tris[:, 1]] - p0
        e2 = self.x[self.tris[:, 2]] - p0
        # current edges expressed in the frozen rest basis -> (T,2,2)
        Ac = np.stack((np.einsum("ti,tki->tk", e1, self.basis),
                       np.einsum("ti,tki->tk", e2, self.basis)), axis=1)
        F = Ac @ self.Ar_inv                      # material-coordinate gradient
        G = np.zeros((self.n, 2, 2))
        for t, tri in enumerate(self.tris):
            for vidx in tri:
                G[vidx] += F[t]
        G /= self.tri_counts[:, None, None]
        return G

    def _contact_distances(self, xc: float) -> np.ndarray:
        x = self.x
        sd_g, _ = _box_sdf_and_normal(x, self.ground_c, self.ground_h)
        sd_l, _ = _box_sdf_and_normal(x, np.array([-xc, 0.0, 1.3]), self.wall_h)
        sd_r, _ = _box_sdf_and_normal(x, np.array([xc, 0.0, 1.3]), self.wall_h)
        sd = np.minimum(np.minimum(sd_g, sd_l), sd_r)
        # self-contact gaps (nearest non-self vertex, minus activation radius)
        dvec = x[:, None, :] - x[None, :, :]
        d = np.sqrt(np.einsum("ijk,ijk->ij", dvec, dvec))
        np.fill_diagonal(d, np.inf)
        sd_self = d.min(axis=1) - self.r_self
        return np.minimum(sd, sd_self)

    def _frame(self, time: float) -> StateFrame:
        xc = self._wall_xc(time)
        sd = self._contact_distances(xc)
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradients(),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": self.ext_force.copy(),
                "contact_signed_distance": sd,
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        # fixed-substep march to the requested sample time
        guard = 0
        while self.t < time - 1e-9 and guard < 10 * self.substeps + 10:
            h = min(self.dt, time - self.t)
            old = self.dt
            self.dt = h
            self._step(self.t)
            self.dt = old
            guard += 1
        self.t = max(self.t, time)
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "mass-spring thin sheet with projection contact",
            "substeps_per_sample": self.substeps,
            "spring_counts": int(len(self.sp_a)),
            "self_contact_radius": self.r_self,
        }
