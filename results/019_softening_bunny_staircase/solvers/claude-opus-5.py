"""Work-softening viscoplastic FEM solver for the softening bunny staircase.

Model
-----
The frozen bunny surface mesh is embedded in a conforming tetrahedral lattice
obtained by voxelising the closed mesh (five tets per cell, checkerboard split
so the diagonals of shared faces match).  The lattice is advanced with an
explicit symplectic-Euler linear-tetrahedron finite element method:

* corotational elastic stress evaluated on the elastic part of the
  multiplicative decomposition ``F = Fe @ Fp``,
* Kelvin-Voigt viscous stress (viscoplastic flow, frozen ``damping``),
* gravity, and penalty + Coulomb-friction contact against the frozen static
  boxes (ground plus the four stair slabs) using the frozen
  ``restitution``/``friction``/``contact_length_scale``.

Plasticity is a von-Mises return mapping in principal Hencky-strain space with
volume-preserving plastic flow (``det Fp == 1``) and *work softening*: the
yield strain and the elastic moduli decay with the accumulated equivalent
plastic strain (frozen ``softening_rate``), so repeatedly impacted regions
become soft, deform permanently and droop over the step edges.

Everything is produced by actual time stepping; no keyframes or scripted
motion are used anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# Cube corner ids: id = 4*dx + 2*dy + dz.
_CORNER = np.array(
    [[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], dtype=np.int64
)
# Five-tet split of a cube (central tet uses the alternating corner set).
_TETS_A = np.array(
    [[0, 1, 3, 5], [0, 3, 2, 6], [0, 5, 6, 4], [3, 5, 6, 7], [0, 3, 5, 6]], dtype=np.int64
)
# Mirrored split (x-flip: corner id XOR 4) so neighbouring cells stay conforming.
_TETS_B = _TETS_A ^ 4

_FALLBACK_OBSTACLES = (
    ((0.0, 0.0, -0.04), (3.0, 1.6, 0.04)),
    ((-0.75, 0.0, 0.96), (0.36, 1.05, 0.14)),
    ((-0.15, 0.0, 0.68), (0.36, 1.05, 0.14)),
    ((0.45, 0.0, 0.40), (0.36, 1.05, 0.14)),
    ((1.05, 0.0, 0.12), (0.36, 1.05, 0.14)),
)
def _inside_columns(vertices, faces, gx, gy, gz):
    """Ray-cast parity test for every lattice node of the grid ``gx x gy x gz``.

    Casts a +z ray from each (x, y) column and counts triangle crossings above
    each node height.  Returns a boolean array of shape ``(nx, ny, nz)``.
    """
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    good = np.abs(det) > 1e-14
    inv_det = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)

    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    px = xx.reshape(-1)
    py = yy.reshape(-1)

    rx = px[:, None] - p0[None, :, 0]
    ry = py[:, None] - p0[None, :, 1]
    u = (rx * e2[None, :, 1] - ry * e2[None, :, 0]) * inv_det[None, :]
    v = (ry * e1[None, :, 0] - rx * e1[None, :, 1]) * inv_det[None, :]
    hit = good[None, :] & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    z_hit = p0[None, :, 2] + u * e1[None, :, 2] + v * e2[None, :, 2]
    z_hit = np.where(hit, z_hit, -np.inf)

    inside = np.zeros((px.size, gz.size), dtype=bool)
    for k, z in enumerate(gz):
        inside[:, k] = (np.count_nonzero(z_hit > z, axis=1) % 2) == 1
    return inside.reshape(gx.size, gy.size, gz.size)


def _bary(points, a, b, c, d):
    """Barycentric coordinates of ``points`` in the tets ``(a, b, c, d)``."""
    m = np.stack([b - a, c - a, d - a], axis=-1)
    rhs = (points - a)[..., None]
    try:
        sol = np.linalg.solve(m, rhs)[..., 0]
    except np.linalg.LinAlgError:
        sol = np.einsum("...ij,...j->...i", np.linalg.pinv(m), (points - a))
    w = np.empty(sol.shape[:-1] + (4,), dtype=float)
    w[..., 1:] = sol
    w[..., 0] = 1.0 - sol.sum(axis=-1)
    return w


def _build_lattice(vertices, faces, target_cells=950):
    """Voxelise the closed surface mesh into a conforming tet lattice."""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    extent = hi - lo
    box_vol = float(np.prod(np.maximum(extent, 1e-6)))
    h = float(np.clip((box_vol / max(target_cells, 1)) ** (1.0 / 3.0), 0.03, 0.12))

    # Irrational offset keeps lattice nodes off mesh vertices/edges.
    origin = lo - h * 0.6180339887498949
    dims = np.maximum(np.ceil((hi - origin) / h).astype(np.int64) + 2, 2)
    nx, ny, nz = (int(d) for d in dims)
    gx = origin[0] + h * np.arange(nx)
    gy = origin[1] + h * np.arange(ny)
    gz = origin[2] + h * np.arange(nz)

    inside = _inside_columns(vertices, faces, gx, gy, gz)

    cell = np.zeros((nx - 1, ny - 1, nz - 1), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                cell |= inside[dx : nx - 1 + dx, dy : ny - 1 + dy, dz : nz - 1 + dz]

    # Guarantee every surface vertex has a host cell (thin ears, etc.).
    vidx = np.floor((vertices - origin) / h).astype(np.int64)
    vidx = np.clip(vidx, 0, np.array([nx - 2, ny - 2, nz - 2], dtype=np.int64))
    cell[vidx[:, 0], vidx[:, 1], vidx[:, 2]] = True

    ci, cj, ck = np.nonzero(cell)
    if ci.size == 0:  # pragma: no cover - degenerate safeguard
        raise RuntimeError("empty voxelisation of subject mesh")

    node_id = -np.ones((nx, ny, nz), dtype=np.int64)
    corners = np.stack([ci, cj, ck], axis=1)[:, None, :] + _CORNER[None, :, :]
    flat = corners.reshape(-1, 3)
    node_id[flat[:, 0], flat[:, 1], flat[:, 2]] = 0
    ni, nj, nk = np.nonzero(node_id == 0)
    node_id[ni, nj, nk] = np.arange(ni.size)
    nodes = np.stack([gx[ni], gy[nj], gz[nk]], axis=1)

    cell_nodes = node_id[corners[:, :, 0], corners[:, :, 1], corners[:, :, 2]]
    parity = ((ci + cj + ck) % 2).astype(bool)
    pattern = np.where(parity[:, None, None], _TETS_B[None], _TETS_A[None])
    tets = np.take_along_axis(
        np.repeat(cell_nodes[:, None, :], 5, axis=1), pattern, axis=2
    ).reshape(-1, 4)

    # Enforce positive orientation.
    d0 = nodes[tets[:, 1]] - nodes[tets[:, 0]]
    d1 = nodes[tets[:, 2]] - nodes[tets[:, 0]]
    d2 = nodes[tets[:, 3]] - nodes[tets[:, 0]]
    vol6 = np.einsum("ij,ij->i", d0, np.cross(d1, d2))
    flip = vol6 < 0.0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]

    cell_lookup = -np.ones((nx - 1, ny - 1, nz - 1), dtype=np.int64)
    cell_lookup[ci, cj, ck] = np.arange(ci.size)
    host_cell = cell_lookup[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
    cand = host_cell[:, None] * 5 + np.arange(5)[None, :]
    cand_tets = tets[cand.reshape(-1)].reshape(-1, 5, 4)
    pts = np.repeat(vertices[:, None, :], 5, axis=1)
    w = _bary(
        pts,
        nodes[cand_tets[:, :, 0]],
        nodes[cand_tets[:, :, 1]],
        nodes[cand_tets[:, :, 2]],
        nodes[cand_tets[:, :, 3]],
    )
    best = np.argmax(w.min(axis=2), axis=1)
    rows = np.arange(vertices.shape[0])
    embed_tet = cand[rows, best]
    embed_nodes = cand_tets[rows, best]
    embed_w = w[rows, best]
    embed_w = embed_w / np.clip(embed_w.sum(axis=1, keepdims=True), 1e-12, None)
    return nodes, tets, embed_nodes, embed_w, embed_tet, h
def _load_obstacles(scene):
    """Read the frozen static collision boxes from ``scene_geometry.yaml``."""
    boxes = []
    try:
        import yaml  # type: ignore

        root = Path(__file__).resolve().parent.parent
        with open(root / "scene_geometry.yaml", "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for obj in data.get("objects", []) or []:
            geom = obj.get("geometry", {}) or {}
            if str(geom.get("type", "")) != "box":
                continue
            if not bool(obj.get("collision_enabled", True)):
                continue
            half = np.asarray(geom.get("half_extents", [0.0, 0.0, 0.0]), dtype=float)
            trans = np.asarray(
                (obj.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0]), dtype=float
            )
            if half.shape == (3,) and trans.shape == (3,) and np.all(half > 0.0):
                boxes.append((trans, half))
    except Exception:
        boxes = []
    if not boxes:
        boxes = [
            (np.asarray(c, dtype=float), np.asarray(h, dtype=float))
            for c, h in _FALLBACK_OBSTACLES
        ]
    centers = np.stack([b[0] for b in boxes], axis=0)
    halves = np.stack([b[1] for b in boxes], axis=0)
    return centers, halves


def _box_sdf(points, centers, halves):
    """Signed distance and outward normal to a set of axis-aligned boxes.

    Returns ``(gap, normal)`` for the closest box per point; ``gap < 0`` means
    penetration and ``normal`` points out of the surface.
    """
    d = points[:, None, :] - centers[None, :, :]
    q = np.abs(d) - halves[None, :, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=2)
    inside = np.minimum(np.max(q, axis=2), 0.0)
    dist = outside + inside  # [P, B]
    best = np.argmin(dist, axis=1)
    rows = np.arange(points.shape[0])
    gap = dist[rows, best]
    qb = q[rows, best]
    db = d[rows, best]
    sgn = np.where(db >= 0.0, 1.0, -1.0)
    normal = np.where(qb[:, :] > 0.0, qb, 0.0) * sgn
    nrm = np.linalg.norm(normal, axis=1)
    # Inside the box: push out along the least-penetrating axis.
    deep = nrm < 1e-12
    if np.any(deep):
        axis = np.argmax(qb[deep], axis=1)
        fix = np.zeros((int(np.count_nonzero(deep)), 3))
        fix[np.arange(fix.shape[0]), axis] = sgn[deep][np.arange(fix.shape[0]), axis]
        normal[deep] = fix
        nrm = np.linalg.norm(normal, axis=1)
    normal = normal / np.clip(nrm, 1e-12, None)[:, None]
    return gap, normal
class BenchmarkSolver:
    """Explicit viscoplastic FEM solver with work softening and penalty contact."""

    MAX_DT = 4.0e-4

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self.time = 0.0

        self.surface = np.asarray(scene.vertices, dtype=float)
        asset = Path(__file__).resolve().parent.parent / cfg["subject_asset"]
        with np.load(asset) as data:
            self.faces = np.asarray(data["faces"], dtype=np.int64)
        self.n_surf = int(self.surface.shape[0])

        nodes, tets, e_nodes, e_w, e_tet, h = _build_lattice(self.surface, self.faces)
        self.rest_nodes = nodes
        self.tets = tets
        self.embed_nodes = e_nodes
        self.embed_w = e_w
        self.embed_tet = e_tet
        self.cell_size = h
        self.n_node = int(nodes.shape[0])
        self.n_tet = int(tets.shape[0])

        # Rest shape matrices and volumes.
        x0 = nodes[tets]
        dm = np.stack(
            [x0[:, 1] - x0[:, 0], x0[:, 2] - x0[:, 0], x0[:, 3] - x0[:, 0]], axis=2
        )
        det = np.linalg.det(dm)
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        self.vol = np.abs(det) / 6.0
        self.dm_inv = np.linalg.inv(dm)
        rest_volume = float(self.vol.sum())

        self.total_mass = float(cfg["subject_mass"])
        self.density = self.total_mass / max(rest_volume, 1e-9)
        node_mass = np.zeros(self.n_node)
        lumped = self.density * self.vol * 0.25
        for c in range(4):
            node_mass += np.bincount(tets[:, c], weights=lumped, minlength=self.n_node)
        # Floor tied to the median lumped mass: keeps the explicit contact and
        # elastic stability limits uniform across sliver-adjacent nodes.
        m_med = float(np.median(node_mass[node_mass > 0.0])) if np.any(node_mass > 0.0) else 1.0
        self.node_mass = np.maximum(node_mass, 0.25 * m_med)
        self.inv_mass = 1.0 / self.node_mass

        # Material: the frozen ``stiffness`` sets the elastic modulus scale of a
        # very soft, clay-like viscoplastic solid.
        self.young0 = 55.0 * float(cfg["stiffness"])
        nu = 0.4
        self.mu0 = self.young0 / (2.0 * (1.0 + nu))
        self.lam0 = self.young0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.eta = 4.0 * float(cfg["damping"])
        self.mass_damping = 0.02
        self.yield0 = 0.045
        self.soften = float(cfg["softening_rate"])
        self.yield_floor = 0.18
        self.stiff_floor = 0.22
        self.plastic_relax = 0.55

        # Contact parameters.
        self.contact_scale = float(cfg["contact_length_scale"])
        self.mu_fric = float(cfg["friction"])
        rest = float(np.clip(cfg["restitution"], 1e-3, 0.98))
        self.k_contact = 1200.0
        m_ref = self.total_mass / max(self.n_surf, 1)
        log_e = np.log(rest)
        zeta = -log_e / np.sqrt(log_e * log_e + np.pi * np.pi)
        c_target = 2.0 * zeta * np.sqrt(self.k_contact * m_ref)
        # Explicit stability: keep the penalty dashpot under 2 m / dt.
        m_min = float(np.min(self.node_mass))
        self.c_contact = float(min(c_target, 0.6 * m_min / self.MAX_DT))
        self.contact_offset = 0.35 * self.contact_scale
        self.force_cap = 20.0
        self.centers, self.halves = _load_obstacles(scene)

        # State.
        self.x = nodes.copy()
        v0 = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0, (self.n_node, 1))
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.grav_force = self.node_mass[:, None] * self.gravity[None, :]
        self.G = np.tile(np.eye(3), (self.n_tet, 1, 1))  # inverse plastic part
        self.alpha = np.zeros(self.n_tet)  # accumulated equivalent plastic strain
        self.plastic_work = 0.0
        self.contact_force_surf = np.zeros((self.n_surf, 3))
        self.n_steps = 0
        self.surf_mass = np.full(self.n_surf, self.total_mass / self.n_surf)
        self.vert_rest_vol = rest_volume / self.n_surf
        self._frame = None
    # -- embedding helpers -------------------------------------------------
    def _surface_state(self):
        p = np.einsum("vc,vcd->vd", self.embed_w, self.x[self.embed_nodes])
        u = np.einsum("vc,vcd->vd", self.embed_w, self.v[self.embed_nodes])
        return p, u

    def _scatter(self, force_surf, out):
        w = self.embed_w
        idx = self.embed_nodes.reshape(-1)
        for d in range(3):
            contrib = (w * force_surf[:, d : d + 1]).reshape(-1)
            out[:, d] += np.bincount(idx, weights=contrib, minlength=self.n_node)
        return out

    # -- contact -----------------------------------------------------------
    def _contact_force(self, p_surf, u_surf):
        gap, normal = _box_sdf(p_surf, self.centers, self.halves)
        pen = self.contact_offset - gap
        active = pen > 0.0
        force = np.zeros_like(p_surf)
        if not np.any(active):
            return force, gap
        n = normal[active]
        d = pen[active]
        vel = u_surf[active]
        vn = np.einsum("ij,ij->i", vel, n)
        fn = self.k_contact * d - self.c_contact * vn
        fn = np.maximum(fn, 0.0)
        fn = np.minimum(fn, self.force_cap)
        vt = vel - vn[:, None] * n
        vt_mag = np.linalg.norm(vt, axis=1)
        # Regularised Coulomb friction (softened where the material has yielded).
        ft_mag = np.minimum(self.mu_fric * fn, self.mu_fric * fn * vt_mag / 0.05)
        t_dir = vt / np.clip(vt_mag, 1e-9, None)[:, None]
        force[active] = fn[:, None] * n - ft_mag[:, None] * t_dir
        return force, gap

    # -- constitutive update ----------------------------------------------
    def _internal_force(self, dt):
        tets = self.tets
        x = self.x[tets]
        ds = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
        F = ds @ self.dm_inv
        Fe = F @ self.G

        U, s, Vt = np.linalg.svd(Fe)
        # Handle reflected elements so the corotational stress stays restoring.
        detU = np.linalg.det(U)
        detV = np.linalg.det(Vt)
        flipU = detU < 0.0
        if np.any(flipU):
            U[flipU, :, 2] *= -1.0
            s[flipU, 2] *= -1.0
        flipV = detV < 0.0
        if np.any(flipV):
            Vt[flipV, 2, :] *= -1.0
            s[flipV, 2] *= -1.0
        s = np.clip(s, 0.06, 6.0)

        eps = np.log(s)
        tr = eps.sum(axis=1) / 3.0
        dev = eps - tr[:, None]
        dev_norm = np.linalg.norm(dev, axis=1)

        # Work softening: yield strain decays with accumulated plastic strain.
        soft = 1.0 / (1.0 + self.soften * self.alpha * 6.0)
        y = self.yield0 * np.maximum(soft, self.yield_floor)

        over = dev_norm - y
        yielding = over > 0.0
        if np.any(yielding):
            rate = min(1.0, self.plastic_relax)
            scale = np.ones_like(dev_norm)
            safe = np.clip(dev_norm, 1e-12, None)
            scale[yielding] = 1.0 - rate * over[yielding] / safe[yielding]
            d_eps = dev * (scale - 1.0)[:, None]
            dgamma = np.linalg.norm(d_eps, axis=1)
            self.alpha += dgamma
            mu_eff_old = self.mu0 * np.maximum(
                1.0 / (1.0 + self.soften * self.alpha), self.stiff_floor
            )
            self.plastic_work += float(
                np.sum(2.0 * mu_eff_old * dgamma * dev_norm * self.vol)
            )
            # Plastic flow is volume preserving: push the excess into G = Fp^-1
            # so that Fe_new = U exp(eps + d_eps) Vt = F @ (G @ V exp(d_eps) V^T).
            factor = np.exp(d_eps)
            M = np.einsum("nji,nj,njk->nik", Vt, factor, Vt)
            self.G = self.G @ M
            dg = np.linalg.det(self.G)
            dg = np.where(np.abs(dg) < 1e-9, 1e-9, dg)
            self.G *= np.cbrt(1.0 / dg)[:, None, None]
            eps = eps + d_eps
            s = np.exp(np.clip(eps, -2.8, 1.8))
            Fe = np.einsum("nij,nj,njk->nik", U, s, Vt)

        self.alpha = np.clip(self.alpha, 0.0, 12.0)
        knock = np.maximum(1.0 / (1.0 + self.soften * self.alpha), self.stiff_floor)
        mu = self.mu0 * knock
        lam = self.lam0 * knock

        R = np.einsum("nij,njk->nik", U, Vt)
        J = np.prod(s, axis=1)
        # Corotational (co-rotated linear) first Piola-Kirchhoff stress.
        P = 2.0 * mu[:, None, None] * (Fe - R) + (lam * (J - 1.0))[:, None, None] * R

        # Kelvin-Voigt viscosity in material coordinates.
        vv = self.v[tets]
        dv = np.stack([vv[:, 1] - vv[:, 0], vv[:, 2] - vv[:, 0], vv[:, 3] - vv[:, 0]], axis=2)
        Fdot = dv @ self.dm_inv
        P = P + self.eta * Fdot

        H = -np.einsum("n,nij,nkj->nik", self.vol, P, self.dm_inv)
        force = np.zeros((self.n_node, 3))
        for c in range(3):
            for d in range(3):
                force[:, d] += np.bincount(
                    tets[:, c + 1], weights=H[:, d, c], minlength=self.n_node
                )
        f0 = -H.sum(axis=2)
        for d in range(3):
            force[:, d] += np.bincount(tets[:, 0], weights=f0[:, d], minlength=self.n_node)
        return force, F
    # -- time stepping -----------------------------------------------------
    def _step(self, dt):
        p_surf, u_surf = self._surface_state()
        f_contact_surf, _ = self._contact_force(p_surf, u_surf)
        self.contact_force_surf = f_contact_surf

        force, _ = self._internal_force(dt)
        force += self.grav_force
        force -= (self.mass_damping * self.node_mass)[:, None] * self.v
        self._scatter(f_contact_surf, force)

        self.v += dt * force * self.inv_mass[:, None]
        speed = np.linalg.norm(self.v, axis=1)
        hot = speed > 25.0
        if np.any(hot):
            self.v[hot] *= (25.0 / speed[hot])[:, None]
        self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        self.x += dt * self.v
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.x, -12.0, 12.0, out=self.x)
        self.n_steps += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1e-12:
            span = target - self.time
            n_sub = max(int(np.ceil(span / self.MAX_DT)), 1)
            dt = span / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self.time = target
        self._frame = self._make_frame(target)
        return self._frame

    def _make_frame(self, time: float) -> StateFrame:
        p_surf, u_surf = self._surface_state()
        gap, _ = _box_sdf(p_surf, self.centers, self.halves)

        # Total deformation gradient of the host element of each surface vertex.
        x = self.x[self.tets]
        ds = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
        F_tet = ds @ self.dm_inv
        F_vert = F_tet[self.embed_tet]
        F_vert = np.nan_to_num(F_vert, nan=1.0, posinf=1.0, neginf=-1.0)
        F_vert = np.clip(F_vert, -20.0, 20.0)

        J = np.clip(np.linalg.det(F_vert), 1e-4, 1e4)
        vol_vert = self.vert_rest_vol * J
        alpha_vert = self.alpha[self.embed_tet]

        contact = np.nan_to_num(self.contact_force_surf, nan=0.0, posinf=0.0, neginf=0.0)
        gap = np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)

        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(np.nan_to_num(p_surf)),
            velocity=np.ascontiguousarray(np.nan_to_num(u_surf)),
            mass=self.surf_mass.copy(),
            body_id=np.zeros(self.n_surf, dtype=np.int64),
            attributes={
                "mass_quantity": self.surf_mass.copy(),
                "deformation_gradient": np.ascontiguousarray(F_vert),
                "volume_quantity": vol_vert,
                "plastic_strain": alpha_vert,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.surf_mass[:, None] * self.gravity[None, :],
                "interaction_force": np.ascontiguousarray(contact),
                "contact_signed_distance": np.ascontiguousarray(gap),
            },
            scalars={
                "contact_length_scale": float(self.contact_scale),
                "characteristic_length": float(self.cell_size),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        det = float("nan")
        try:
            x = self.x[self.tets]
            ds = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
            det = float(np.median(np.linalg.det(ds @ self.dm_inv)))
        except Exception:
            pass
        return {
            "method": "embedded_lattice_corotational_FEM_von_mises_work_softening",
            "nodes": int(self.n_node),
            "elements": int(self.n_tet),
            "surface_vertices": int(self.n_surf),
            "cell_size": float(self.cell_size),
            "steps": int(self.n_steps),
            "sim_time": float(self.time),
            "young_modulus": float(self.young0),
            "viscosity": float(self.eta),
            "max_plastic_strain": float(np.max(self.alpha)) if self.n_tet else 0.0,
            "mean_plastic_strain": float(np.mean(self.alpha)) if self.n_tet else 0.0,
            "plastic_work": float(self.plastic_work),
            "median_det_F": det,
            "total_mass": float(self.total_mass),
        }

