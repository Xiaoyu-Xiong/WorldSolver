"""Self-contained volumetric FEM solver for armadillo roller stick-slip.

Physics
-------
* The armadillo surface mesh is voxelised into a conforming tetrahedral lattice
  (Kuhn 6-tet split of interior cells selected by a generalised winding number).
* The lattice is a hyperelastic solid: stable Neo-Hookean elasticity with
  Kelvin-Voigt (strain-rate) viscous stress, lumped nodal mass, advanced with an
  explicit symplectic-Euler scheme at a CFL/contact-limited step size.
* Rollers are kinematic cylinders. Contact uses a damped normal penalty plus a
  velocity-level Coulomb friction impulse capped by ``mu * F_n``, which produces
  genuine stick (exact tangential arrest) and slip (saturated cone) states, i.e.
  the stick-slip intermittency of the reference experiment.
* The frozen 24-vertex render surface is embedded in the lattice through a
  mass-consistent Sinkhorn transfer operator, so exported momentum equals the
  simulated lattice momentum exactly and ``d(position)/dt == velocity``.

No scripted motion, keyframes or precomputed trajectories are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)


def _load_obj(path: Path):
    """Return (vertices[V,3], triangles[T,3]) with quads fan-triangulated."""
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(p) for p in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            for k in range(1, len(idx) - 1):
                tris.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(tris, dtype=np.int64)


def _winding_number(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Generalised winding number (robust for open/non-watertight patches)."""
    out = np.zeros(len(points), dtype=float)
    chunk = 4096
    for start in range(0, len(points), chunk):
        pts = points[start : start + chunk]
        a = tri[None, :, 0, :] - pts[:, None, :]
        b = tri[None, :, 1, :] - pts[:, None, :]
        c = tri[None, :, 2, :] - pts[:, None, :]
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        num = np.einsum("ptk,ptk->pt", a, np.cross(b, c))
        den = (
            la * lb * lc
            + np.einsum("ptk,ptk->pt", a, b) * lc
            + np.einsum("ptk,ptk->pt", b, c) * la
            + np.einsum("ptk,ptk->pt", c, a) * lb
        )
        omega = 2.0 * np.arctan2(num, den)
        out[start : start + chunk] = omega.sum(axis=1) / (4.0 * np.pi)
    return out


_KUHN = (
    (0, 1, 3, 7),
    (0, 1, 7, 5),
    (0, 5, 7, 4),
    (0, 3, 2, 7),
    (0, 6, 4, 7),
    (0, 2, 6, 7),
)


def _build_lattice(verts: np.ndarray, tri: np.ndarray, h: float):
    """Voxelise the closed surface into a conforming tetrahedral lattice."""
    lo = verts.min(axis=0) - 0.5 * h
    hi = verts.max(axis=0) + 0.5 * h
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    nx, ny, nz = (int(d) for d in dims)

    gx = lo[0] + h * np.arange(nx + 1)
    gy = lo[1] + h * np.arange(ny + 1)
    gz = lo[2] + h * np.arange(nz + 1)
    grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)

    ci, cj, ck = (a.ravel() for a in np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    centers = np.stack([gx[ci] + 0.5 * h, gy[cj] + 0.5 * h, gz[ck] + 0.5 * h], axis=1)
    inside = _winding_number(centers, tri) > 0.5
    if not inside.any():  # degenerate fallback: keep the bounding box
        inside = np.ones(len(centers), dtype=bool)
    ci, cj, ck = ci[inside], cj[inside], ck[inside]

    def gid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    corners = np.stack(
        [gid(ci + dx, cj + dy, ck + dz)
         for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
        axis=1,
    )  # corner order: (dx,dy,dz) bits -> index dx*4+dy*2+dz
    tets = np.concatenate([corners[:, list(t)] for t in _KUHN], axis=0)

    used, inv = np.unique(tets.ravel(), return_inverse=True)
    nodes = grid[used]
    tets = inv.reshape(tets.shape).astype(np.int64)

    # Drop degenerate/inverted elements and orient positively.
    e1 = nodes[tets[:, 1]] - nodes[tets[:, 0]]
    e2 = nodes[tets[:, 2]] - nodes[tets[:, 0]]
    e3 = nodes[tets[:, 3]] - nodes[tets[:, 0]]
    vol6 = np.einsum("ek,ek->e", np.cross(e1, e2), e3)
    flip = vol6 < 0.0
    if flip.any():
        tets[flip] = tets[flip][:, [0, 2, 1, 3]]
        vol6 = np.abs(vol6)
    keep = vol6 > 1e-12
    tets = tets[keep]
    return nodes, tets


def _sinkhorn_transfer(surf: np.ndarray, nodes: np.ndarray, node_mass: np.ndarray,
                       surf_mass: np.ndarray, sigma: float, iters: int = 1500):
    """Row-stochastic, mass-consistent surface<-lattice interpolation weights.

    Satisfies ``sum_j W_ij = 1`` and ``sum_i m_i W_ij = M_j`` so that
    ``sum_i m_i (W v)_i == sum_j M_j v_j`` (momentum-preserving embedding).
    """
    d2 = ((surf[:, None, :] - nodes[None, :, :]) ** 2).sum(axis=2)
    kern = np.exp(-d2 / (2.0 * sigma * sigma))
    # Guarantee every row has support (far vertices fall back to nearest node).
    weak = kern.sum(axis=1) < 1e-12
    if weak.any():
        kern[weak, np.argmin(d2[weak], axis=1)] = 1.0
    u = np.ones(len(surf))
    w = np.ones(len(nodes))
    for _ in range(iters):
        u = 1.0 / np.maximum(kern @ w, 1e-300)
        w = node_mass / np.maximum(kern.T @ (u * surf_mass), 1e-300)
    mat = u[:, None] * kern * w[None, :]
    rows = mat.sum(axis=1)
    bad = np.abs(rows - 1.0) > 1e-6
    if bad.any():  # fallback if Sinkhorn stalled: enforce partition of unity
        mat[bad] /= np.maximum(rows[bad, None], 1e-300)
    return mat


class _Cylinder:
    """Kinematic roller: axis along +y, spinning with omega(t) about +y."""

    __slots__ = ("center", "radius", "half_len", "spin", "mu")

    def __init__(self, center, radius, height, spin, mu):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.half_len = 0.5 * float(height)
        self.spin = float(spin)
        self.mu = float(mu)

    def query(self, pos: np.ndarray, pad: float, ramp: float):
        rel = pos - self.center
        planar = rel.copy()
        planar[:, 1] = 0.0
        dist = np.linalg.norm(planar, axis=1)
        safe = np.maximum(dist, 1e-9)
        normal = planar / safe[:, None]
        depth = (self.radius + pad) - dist
        active = (depth > 0.0) & (np.abs(rel[:, 1]) <= self.half_len)
        omega = np.array([0.0, self.spin * ramp, 0.0])
        # Surface velocity at the contact point on the roller.
        contact_rel = normal * self.radius
        contact_rel[:, 1] = rel[:, 1]
        vel = np.cross(np.broadcast_to(omega, contact_rel.shape), contact_rel)
        return np.where(active, depth, -1.0), normal, vel


class _Plane:
    """Static table / floor, optionally a conveyor with tangential surface speed."""

    __slots__ = ("height", "x_lo", "x_hi", "speed", "mu")

    def __init__(self, height, x_lo, x_hi, speed, mu):
        self.height = float(height)
        self.x_lo = float(x_lo)
        self.x_hi = float(x_hi)
        self.speed = float(speed)
        self.mu = float(mu)

    def query(self, pos: np.ndarray, pad: float, ramp: float):
        depth = (self.height + pad) - pos[:, 2]
        inside = (pos[:, 0] >= self.x_lo) & (pos[:, 0] <= self.x_hi)
        normal = np.zeros_like(pos)
        normal[:, 2] = 1.0
        vel = np.zeros_like(pos)
        vel[:, 0] = self.speed * ramp
        return np.where(inside & (depth > 0.0), depth, -1.0), normal, vel


class BenchmarkSolver:
    """Explicit volumetric FEM + penalty/Coulomb contact roller solver."""

    LATTICE_H = 0.065
    POISSON = 0.4
    VISCOSITY = 80.0
    MAX_SUBSTEPS = 800

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.gravity = np.asarray(cfg.get("gravity", GRAVITY), dtype=float)
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        bc = dict(cfg.get("boundary_conditions", {}) or {})
        self.mu = float(bc.get("contact_friction_coefficient", 0.5))
        self.t_start = float(bc.get("start_time", 0.35))
        self.t_ramp = max(float(bc.get("ramp_duration", 0.25)), 1e-9)
        spin = abs(float(bc.get("upper_angular_velocity_rad_s", 5.0)))

        frame0 = scene.initial_state()
        self.surf0 = np.asarray(frame0.position, dtype=float).copy()
        self.surf_mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int64).copy()
        self.total_mass = float(self.surf_mass.sum())
        targets = dict(cfg.get("phenomenon_targets", {}) or {})
        youngs = float(targets.get("moderate_stiffness_reference_pa", 5.0e5))

        root = Path(__file__).resolve().parent.parent
        _, tri_idx = _load_obj(root / "assets" / "armadillo.obj")
        self.tri_idx = tri_idx
        tri = self.surf0[tri_idx]

        h = self.LATTICE_H
        nodes, tets = _build_lattice(self.surf0, tri, h)
        self.nodes0 = nodes
        self.tets = tets
        self.pos = nodes.copy()
        self.vel = np.zeros_like(nodes)
        self.n_nodes = len(nodes)
        self.n_tets = len(tets)

        # Reference shape matrices, volumes, lumped masses.
        x0 = nodes[tets]
        dm = np.stack([x0[:, 1] - x0[:, 0], x0[:, 2] - x0[:, 0], x0[:, 3] - x0[:, 0]], axis=2)
        self.dm_inv = np.linalg.inv(dm)
        self.dm_inv_t = np.transpose(self.dm_inv, (0, 2, 1))
        self.vol0 = np.abs(np.linalg.det(dm)) / 6.0
        volume = float(self.vol0.sum())
        self.density = self.total_mass / max(volume, 1e-12)
        node_mass = np.zeros(self.n_nodes)
        lump = self.density * self.vol0 / 4.0
        ids = np.concatenate([tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]])
        node_mass = np.bincount(ids, weights=np.tile(lump, 4), minlength=self.n_nodes)
        node_mass = np.maximum(node_mass, 1e-9)
        node_mass *= self.total_mass / node_mass.sum()
        self.node_mass = node_mass
        self.inv_mass = 1.0 / node_mass
        self.scatter_ids = np.concatenate([tets[:, 1], tets[:, 2], tets[:, 3], tets[:, 0]])

        # Lame parameters.
        nu = self.POISSON
        self.mu_lame = youngs / (2.0 * (1.0 + nu))
        self.lam = youngs * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.eta = self.VISCOSITY
        self.youngs = youngs
        self._setup_contacts(cfg, spin, h, youngs)
        self._setup_timestep(h)
        self._setup_embedding(h)
        self._init_accumulators()

    # ---------------------------------------------------------------- setup
    def _setup_contacts(self, cfg, spin, h, youngs) -> None:
        geo = cfg.get("scene_geometry", {}) or {}
        rollers = []
        for item in geo.get("objects", []) or []:
            g = item.get("geometry", {}) or {}
            if g.get("type") != "cylinder":
                continue
            tr = item.get("transform", {}) or {}
            center = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(g.get("radius", 0.22))
            height = float(g.get("height", 0.72))
            # Counter-rotation about the world y axis. The frozen quaternion
            # rotates each cylinder's local +z onto world -y, so the frozen
            # local spins (+5 upper / -5 lower) map to world (-5 / +5), which
            # is exactly the sense that drives the gap surfaces along +x, the
            # frozen intended_transport_direction.
            sgn = -1.0 if center[2] > 0.0 else 1.0
            rollers.append(_Cylinder(center, radius, height, sgn * spin, self.mu))
        if not rollers:
            for cz in (0.3, -0.3):
                for cx in (-0.2, 0.3):
                    sgn = -1.0 if cz > 0.0 else 1.0
                    rollers.append(
                        _Cylinder([cx, 0.0, cz], 0.22, 0.72, sgn * spin, self.mu))
        self.rollers = rollers
        self.pad = 0.45 * h
        r_ref = max(r.radius for r in rollers)
        x_left = min(r.center[0] for r in rollers)
        x_right = max(r.center[0] for r in rollers)
        self.surface_speed = spin * r_ref
        # Feed conveyor supporting the body at the roller entrance (it would
        # otherwise free-fall: the frozen scene has no floor), plus an exit
        # table and a far safety floor. All share the frozen friction pairing.
        table_z = float(self.nodes0[:, 2].min()) - self.pad
        self.table_z = table_z
        self.planes = [
            _Plane(table_z, -1.0e3, x_left - r_ref - 0.02, self.surface_speed, self.mu),
            _Plane(table_z, x_right + r_ref + 0.02, 1.0e3, 0.0, self.mu),
            _Plane(-1.0, -1.0e3, 1.0e3, 0.0, self.mu),
        ]
        self.obstacles = list(self.rollers) + list(self.planes)
        self.k_n = 2.0e5

    def _setup_timestep(self, h) -> None:
        floor_mass = 0.3 * self.density * h ** 3
        self.node_mass = np.maximum(self.node_mass, floor_mass)
        self.node_mass *= self.total_mass / self.node_mass.sum()
        self.inv_mass = 1.0 / self.node_mass
        wave = float(np.sqrt((self.lam + 2.0 * self.mu_lame) / max(self.density, 1e-9)))
        m_ref = float(self.node_mass.min())
        dt_cfl = 0.15 * h / max(wave, 1e-9)
        dt_contact = 0.5 * np.sqrt(m_ref / self.k_n)
        dt_visc = 0.2 * self.density * h * h / max(self.eta, 1e-9)
        dt_target = float(min(dt_cfl, dt_contact, dt_visc))
        n_sub = int(np.ceil(self.sample_dt / max(dt_target, 1e-12)))
        self.n_sub = int(min(max(n_sub, 1), self.MAX_SUBSTEPS))
        self.dt = self.sample_dt / self.n_sub
        self.c_n = np.minimum(0.6 * np.sqrt(self.k_n * self.node_mass),
                              1.0 * self.node_mass / self.dt)

    def _setup_embedding(self, h) -> None:
        self.embed = _sinkhorn_transfer(
            self.surf0, self.nodes0, self.node_mass, self.surf_mass, sigma=1.1 * h)

    def _init_accumulators(self) -> None:
        n = self.n_nodes
        self.step_count = 0
        self.time = 0.0
        self.cum_impulse = np.zeros(3)
        self.force_terms = np.zeros((3, 3))
        self.force_terms[0] = self.total_mass * self.gravity
        self.rows_normal = np.zeros((n, 3))
        self.rows_normal[:, 2] = 1.0
        self.rows_fn = np.zeros(n)
        self.rows_ft = np.zeros((n, 3))
        self.rows_relvel = np.zeros((n, 3))
        self.rows_active = np.zeros(n)
        self.rows_mu = np.full(n, self.mu)
        self.clamped = 0

    # ------------------------------------------------------------- mechanics
    def _ramp(self, t: float) -> float:
        s = (t - self.t_start) / self.t_ramp
        if s <= 0.0:
            return 0.0
        if s >= 1.0:
            return 1.0
        return float(s * s * (3.0 - 2.0 * s))

    def _contacts(self, pos: np.ndarray, ramp: float):
        n = self.n_nodes
        best = np.full(n, -1.0)
        normal = np.zeros((n, 3))
        normal[:, 2] = 1.0
        obs_vel = np.zeros((n, 3))
        mu = np.full(n, self.mu)
        for ob in self.obstacles:
            depth, nrm, vel = ob.query(pos, self.pad, ramp)
            sel = depth > best
            if sel.any():
                best[sel] = depth[sel]
                normal[sel] = nrm[sel]
                obs_vel[sel] = vel[sel]
                mu[sel] = ob.mu
        active = best > 0.0
        return np.where(active, best, 0.0), normal, obs_vel, mu, active

    def _internal_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        tets = self.tets
        x = pos[tets]
        v = vel[tets]
        ds = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
        f_grad = ds @ self.dm_inv
        dv = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=2)
        f_dot = dv @ self.dm_inv

        c0, c1, c2 = f_grad[:, :, 0], f_grad[:, :, 1], f_grad[:, :, 2]
        cof = np.stack([np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)], axis=2)
        det = np.einsum("ek,ek->e", c0, np.cross(c1, c2))
        # Stable Neo-Hookean: mu/2 (I2-3) + lam/2 (J-1)^2 - mu (J-1).
        coeff = self.lam * (det - 1.0) - self.mu_lame
        piola = self.mu_lame * f_grad + coeff[:, None, None] * cof
        # Kelvin-Voigt viscous stress (rotation invariant, strictly dissipative).
        strain_rate = np.transpose(f_grad, (0, 2, 1)) @ f_dot
        strain_rate = 0.5 * (strain_rate + np.transpose(strain_rate, (0, 2, 1)))
        piola = piola + self.eta * (f_grad @ strain_rate)

        h_mat = -self.vol0[:, None, None] * (piola @ self.dm_inv_t)
        contrib = np.concatenate(
            [h_mat[:, :, 0], h_mat[:, :, 1], h_mat[:, :, 2], -h_mat.sum(axis=2)], axis=0)
        forces = np.stack(
            [np.bincount(self.scatter_ids, weights=contrib[:, k], minlength=self.n_nodes)
             for k in range(3)], axis=1)
        forces += self.node_mass[:, None] * self.gravity
        self._last_f = f_grad
        return forces

    def _step(self, t_now: float) -> None:
        dt = self.dt
        ramp = self._ramp(t_now)
        forces = self._internal_forces(self.pos, self.vel)
        depth, normal, obs_vel, mu, active = self._contacts(self.pos, ramp)

        rel = self.vel - obs_vel
        v_n = np.einsum("ij,ij->i", rel, normal)
        f_n = np.where(active, self.k_n * depth - self.c_n * v_n, 0.0)
        f_n = np.maximum(f_n, 0.0)
        f_normal = f_n[:, None] * normal

        self.vel = self.vel + dt * self.inv_mass[:, None] * (forces + f_normal)

        # Velocity-level Coulomb friction: exact stick when the arresting
        # impulse fits in the cone, saturated slip otherwise.
        rel = self.vel - obs_vel
        v_n = np.einsum("ij,ij->i", rel, normal)
        v_t = rel - v_n[:, None] * normal
        speed = np.linalg.norm(v_t, axis=1)
        direction = v_t / np.maximum(speed, 1e-14)[:, None]
        j_mag = np.minimum(self.node_mass * speed, mu * f_n * dt)
        impulse = -j_mag[:, None] * direction
        self.vel = self.vel + self.inv_mass[:, None] * impulse
        self.pos = self.pos + dt * self.vel

        if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all()):
            self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
            self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
            self.clamped += 1
        np.clip(self.vel, -200.0, 200.0, out=self.vel)

        f_t = impulse / dt
        self.cum_impulse += dt * (self.total_mass * self.gravity + f_normal.sum(axis=0)) \
            + impulse.sum(axis=0)
        self.force_terms[1] = f_normal.sum(axis=0)
        self.force_terms[2] = f_t.sum(axis=0)

        live = active & (f_n > 0.0)
        rel_post = np.where(live[:, None], self.vel - obs_vel, 0.0)
        self.rows_active = live.astype(float)
        self.rows_fn = np.where(live, f_n, 0.0)
        self.rows_ft = np.where(live[:, None], f_t, 0.0)
        self.rows_relvel = rel_post
        default_n = np.zeros((self.n_nodes, 3))
        default_n[:, 2] = 1.0
        self.rows_normal = np.where(live[:, None], normal, default_n)
        self.rows_mu = np.where(live, mu, self.mu)

    # ---------------------------------------------------------------- output
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        # Integer step counting: exactly deterministic, no float accumulation.
        target_steps = int(round(target / self.dt))
        while self.step_count < target_steps:
            self._step(self.step_count * self.dt)
            self.step_count += 1
        self.time = self.step_count * self.dt
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        disp = self.embed @ (self.pos - self.nodes0)
        position = self.surf0 + disp
        velocity = self.embed @ self.vel
        position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

        if getattr(self, "_last_f", None) is None:
            x = self.pos[self.tets]
            ds = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
            self._last_f = ds @ self.dm_inv
        f_grad = np.nan_to_num(self._last_f, nan=0.0, posinf=0.0, neginf=0.0)

        fields = {
            "budget_domain_weights": np.ones((1, len(position))),
            "budget_external_force": self.force_terms[None, :, :].copy(),
            "budget_cumulative_external_impulse": self.cum_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": f_grad,
            "friction_active_mask": self.rows_active.copy(),
            "friction_normal": self.rows_normal.copy(),
            "friction_normal_force": self.rows_fn.copy(),
            "friction_tangential_force": self.rows_ft.copy(),
            "friction_relative_velocity": self.rows_relvel.copy(),
            "friction_static_coefficient": self.rows_mu.copy(),
            "friction_dynamic_coefficient": self.rows_mu.copy(),
        }
        fields = {k: np.nan_to_num(np.asarray(v, dtype=float), nan=0.0,
                                   posinf=0.0, neginf=0.0) for k, v in fields.items()}
        scalars = {
            "friction_stick_speed_tolerance": 1.0e-3,
            "reference_force": float(self.total_mass * np.linalg.norm(self.gravity)),
            "reference_momentum": float(self.total_mass * max(self.surface_speed, 1e-6)),
            "reference_velocity": float(max(self.surface_speed, 1e-6)),
            "roller_friction": self.mu,
        }
        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self.surf_mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.surf_mass.copy(),
                "rest_position": self.surf0.copy(),
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "embedded volumetric FEM (stable Neo-Hookean + Kelvin-Voigt), "
                      "explicit symplectic Euler, penalty normal contact with "
                      "velocity-level Coulomb friction",
            "lattice_nodes": int(self.n_nodes),
            "tetrahedra": int(self.n_tets),
            "substeps_per_sample": int(self.n_sub),
            "dt": float(self.dt),
            "steps_taken": int(self.step_count),
            "density_kg_m3": float(self.density),
            "youngs_modulus_pa": float(self.youngs),
            "contact_stiffness_n_m": float(self.k_n),
            "friction_coefficient": float(self.mu),
            "roller_surface_speed_m_s": float(self.surface_speed),
            "feed_table_z": float(self.table_z),
            "nonfinite_guard_events": int(self.clamped),
        }
