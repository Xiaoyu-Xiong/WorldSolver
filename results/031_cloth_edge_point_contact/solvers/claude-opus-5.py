"""Self-contained cloth solver: square sheet draping on a sharp edge and point.

Physical model
--------------
* Cloth: the frozen 41x41 sheet is discretised as a lumped-mass membrane.
  Structural, shear and bending distance constraints are enforced with a
  heavily substepped position-based (XPBD style) Gauss-Seidel projection over
  vertex-disjoint constraint colourings.  The scheme is unconditionally stable
  for the stiff, nearly inextensible in-plane directions and reproduces cloth
  drape and folding under gravity.
* Contacts: analytic signed-distance fields of the frozen obstacles (ground
  box, sharp edge box, sharp cone read from the scene configuration) with the
  frozen thickness offset.  Collisions are resolved at cloth vertices and, to
  avoid the sharp point or the sharp edge slipping between vertices, also at
  edge midpoints and quad centres, whose corrections are redistributed to the
  supporting vertices.
* Friction: Coulomb friction applied as a position-level tangential clamp
  limited by the normal penetration correction of the substep.
* Self contact: vertex-vertex proximity repulsion from a KD-tree query.

Everything is deterministic (no random numbers, sorted contact pair ordering)
and only NumPy / SciPy are used.
"""
from __future__ import annotations

import math

import numpy as np

try:  # SciPy is optional; self-contact is skipped if unavailable.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover
    _KDTree = None

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Analytic signed distance fields (positive outside, normal points outward)
# ---------------------------------------------------------------------------
def _box_sdf(points, center, half):
    d = points - center
    q = np.abs(d) - half
    qpos = np.maximum(q, 0.0)
    outside = np.sqrt(np.sum(qpos * qpos, axis=1))
    qmax = np.max(q, axis=1)
    phi = outside + np.minimum(qmax, 0.0)
    sgn = np.where(d >= 0.0, 1.0, -1.0)
    grad = sgn * qpos
    glen = np.sqrt(np.sum(grad * grad, axis=1))
    normal = np.zeros_like(d)
    outer = glen > 1e-12
    normal[outer] = grad[outer] / glen[outer, None]
    inner = ~outer
    if np.any(inner):
        axis = np.argmax(q[inner], axis=1)
        block = np.zeros((int(np.count_nonzero(inner)), 3))
        block[np.arange(block.shape[0]), axis] = 1.0
        normal[inner] = block * sgn[inner]
    return phi, normal
def _cone_sdf(points, tip, height, radius):
    """Signed distance to a solid cone: apex at ``tip``, axis along -z."""
    d = points - tip
    rx = d[:, 0]
    ry = d[:, 1]
    r = np.sqrt(rx * rx + ry * ry)
    u = -d[:, 2]  # 0 at the apex, +height at the base disk
    inv_r = np.where(r > 1e-12, 1.0 / np.maximum(r, 1e-12), 0.0)
    ux = rx * inv_r
    uy = ry * inv_r
    # unit radial direction fallback on the axis
    on_axis = r <= 1e-12
    ux = np.where(on_axis, 1.0, ux)
    uy = np.where(on_axis, 0.0, uy)

    slant_len = math.hypot(height, radius)
    s_slant = (r * height - u * radius) / slant_len   # >0 outside lateral face
    s_base = u - height                              # >0 below the base disk
    inside = (s_slant <= 0.0) & (s_base <= 0.0)

    # distance to the lateral segment (0,0)-(radius,height)
    t = (r * radius + u * height) / (slant_len * slant_len)
    t = np.clip(t, 0.0, 1.0)
    cr = t * radius
    cu = t * height
    wr = r - cr
    wu = u - cu
    dist_slant = np.sqrt(wr * wr + wu * wu)
    # distance to the base segment r in [0,radius] at u = height
    br = np.clip(r, 0.0, radius)
    wr2 = r - br
    wu2 = u - height
    dist_base = np.sqrt(wr2 * wr2 + wu2 * wu2)

    use_slant = dist_slant <= dist_base
    dist_out = np.where(use_slant, dist_slant, dist_base)
    nr_out = np.where(use_slant, wr, wr2)
    nu_out = np.where(use_slant, wu, wu2)
    nlen = np.sqrt(nr_out * nr_out + nu_out * nu_out)
    good = nlen > 1e-12
    nr_out = np.where(good, nr_out / np.maximum(nlen, 1e-12), height / slant_len)
    nu_out = np.where(good, nu_out / np.maximum(nlen, 1e-12), -radius / slant_len)

    slant_active = s_slant >= s_base
    nr_in = np.where(slant_active, height / slant_len, 0.0)
    nu_in = np.where(slant_active, -radius / slant_len, 1.0)

    phi = np.where(inside, np.maximum(s_slant, s_base), dist_out)
    nr = np.where(inside, nr_in, nr_out)
    nu = np.where(inside, nu_in, nu_out)
    normal = np.empty_like(d)
    normal[:, 0] = nr * ux
    normal[:, 1] = nr * uy
    normal[:, 2] = -nu
    return phi, normal
class BenchmarkSolver:
    """Substepped position-based cloth solver with analytic obstacle contact."""

    # numerical / material parameters
    SUBSTEPS = 16
    ITERS = 3
    K_STRUCT = 1.0
    K_SHEAR = 0.5
    K_BEND = 0.05
    DAMPING_RATE = 0.5      # 1/s aerodynamic drag on the sheet
    MAX_SPEED = 30.0        # m/s stability clamp
    SELF_ITERS = 1

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        meta = scene.metadata()
        cfg = getattr(scene, "config", {}) or {}
        frame0 = scene.initial_state()

        self.nx, self.ny = (int(v) for v in meta["topology"]["grid_shape"])
        self.n = self.nx * self.ny
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(
            frame0.mass if frame0.mass is not None
            else np.full(self.n, 1.0 / self.n), dtype=float
        )
        self.uv = np.array(
            frame0.attributes.get(
                "uv", np.zeros((self.n, 2))), dtype=float)
        self.gravity = np.asarray(meta["gravity"], dtype=float).reshape(3)
        self.mu = float(meta.get("coulomb_friction", 0.0))
        self.thickness = float(meta.get("cloth_thickness", 0.008))

        cloth_cfg = cfg.get("cloth", {})
        side = float(cloth_cfg.get("side_length", 1.0))
        self.hx = side / max(self.nx - 1, 1)
        self.hy = side / max(self.ny - 1, 1)

        contact_cfg = cfg.get("contact", {}) or {}
        self.offset = float(contact_cfg.get("thickness_offset", self.thickness))
        self.self_radius = max(2.0 * self.offset, 0.6 * min(self.hx, self.hy))
        self.length_scale = float(self.thickness)

        self.obstacles = self._build_obstacles(meta, cfg)
        self.batches = self._build_constraints()
        self._build_aux_samples()
        self._build_self_filter()

        dt_sample = float(cfg.get("sample_dt", 1.0 / 60.0))
        times = list(scene.sample_times())
        if len(times) > 1:
            dt_sample = float(times[1] - times[0])
        self.dt_sample = dt_sample
        self.dt_sub = dt_sample / float(self.SUBSTEPS)
        self.t = 0.0
        self._nsub = 0
        self._min_gap = float("inf")
        self._max_speed = 0.0
    # -- frozen geometry -------------------------------------------------
    def _build_obstacles(self, meta, cfg):
        obs = []
        ground = meta.get("ground", cfg.get("ground"))
        if ground:
            obs.append(
                ("box",
                 np.asarray(ground["center"], dtype=float).reshape(1, 3),
                 np.asarray(ground["half_extents"], dtype=float).reshape(1, 3),
                 None)
            )
        edge = meta.get("sharp_edge", cfg.get("sharp_edge"))
        if edge:
            obs.append(
                ("box",
                 np.asarray(edge["center"], dtype=float).reshape(1, 3),
                 np.asarray(edge["half_extents"], dtype=float).reshape(1, 3),
                 None)
            )
        point = meta.get("sharp_point", cfg.get("sharp_point"))
        if point:
            obs.append(
                ("cone",
                 np.asarray(point["tip"], dtype=float).reshape(1, 3),
                 float(point["height"]),
                 float(point["base_radius"]))
            )
        return obs

    def _sdf(self, points):
        """Minimum signed distance and normal over all frozen obstacles."""
        phi = np.full(points.shape[0], 1e9)
        normal = np.zeros_like(points)
        normal[:, 2] = 1.0
        for kind, a, b, c in self.obstacles:
            if kind == "box":
                p, nrm = _box_sdf(points, a, b)
            else:
                p, nrm = _cone_sdf(points, a, b, c)
            take = p < phi
            phi = np.where(take, p, phi)
            normal[take] = nrm[take]
        return phi, normal
    # -- constraint colouring -------------------------------------------
    def _idx(self, ix, iy):
        return iy * self.nx + ix

    def _pair_batches(self, dx, dy, rest, stiff, mod):
        """Vertex-disjoint batches of distance constraints with stride (dx,dy)."""
        nx, ny = self.nx, self.ny
        out = []
        ax = np.arange(0, nx - dx)
        ay = np.arange(0, ny - dy)
        if ax.size == 0 or ay.size == 0:
            return out
        rx = range(mod) if dx else (0,)
        ry = range(mod) if dy else (0,)
        for cx in rx:
            for cy in ry:
                sx = ax[ax % mod == cx] if dx else ax
                sy = ay[ay % mod == cy] if dy else ay
                if sx.size == 0 or sy.size == 0:
                    continue
                gx, gy = np.meshgrid(sx, sy, indexing="ij")
                gx = gx.ravel()
                gy = gy.ravel()
                i = self._idx(gx, gy).astype(np.int64)
                j = self._idx(gx + dx, gy + dy).astype(np.int64)
                out.append((i, j, float(rest), float(stiff)))
        return out

    def _build_constraints(self):
        hx, hy = self.hx, self.hy
        diag = math.hypot(hx, hy)
        batches = []
        batches += self._pair_batches(1, 0, hx, self.K_STRUCT, 2)
        batches += self._pair_batches(0, 1, hy, self.K_STRUCT, 2)
        # shear: both quad diagonals, coloured on quad parity
        nx, ny = self.nx, self.ny
        for cx in range(2):
            for cy in range(2):
                sx = np.arange(cx, nx - 1, 2)
                sy = np.arange(cy, ny - 1, 2)
                if sx.size == 0 or sy.size == 0:
                    continue
                gx, gy = np.meshgrid(sx, sy, indexing="ij")
                gx = gx.ravel()
                gy = gy.ravel()
                batches.append((self._idx(gx, gy).astype(np.int64),
                                self._idx(gx + 1, gy + 1).astype(np.int64),
                                diag, self.K_SHEAR))
                batches.append((self._idx(gx + 1, gy).astype(np.int64),
                                self._idx(gx, gy + 1).astype(np.int64),
                                diag, self.K_SHEAR))
        # bending: two-cell spans keep the sheet from folding at zero cost
        batches += self._pair_batches(2, 0, 2.0 * hx, self.K_BEND, 4)
        batches += self._pair_batches(0, 2, 2.0 * hy, self.K_BEND, 4)
        return batches
    # -- auxiliary contact samples and self-contact bookkeeping ----------
    def _build_aux_samples(self):
        nx, ny = self.nx, self.ny
        gx, gy = np.meshgrid(np.arange(nx - 1), np.arange(ny), indexing="ij")
        ex = np.stack([self._idx(gx.ravel(), gy.ravel()),
                       self._idx(gx.ravel() + 1, gy.ravel())], axis=1)
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny - 1), indexing="ij")
        ey = np.stack([self._idx(gx.ravel(), gy.ravel()),
                       self._idx(gx.ravel(), gy.ravel() + 1)], axis=1)
        self.aux_edges = np.concatenate([ex, ey], axis=0).astype(np.int64)
        gx, gy = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
        gx = gx.ravel()
        gy = gy.ravel()
        self.aux_quads = np.stack([
            self._idx(gx, gy), self._idx(gx + 1, gy),
            self._idx(gx, gy + 1), self._idx(gx + 1, gy + 1)], axis=1
        ).astype(np.int64)

    def _build_self_filter(self):
        ii = np.arange(self.n)
        self.grid_ix = (ii % self.nx).astype(np.int64)
        self.grid_iy = (ii // self.nx).astype(np.int64)

    # -- position-based projection ---------------------------------------
    def _project_constraints(self, pos):
        for i, j, rest, stiff in self.batches:
            d = pos[j] - pos[i]
            length = np.sqrt(np.sum(d * d, axis=1))
            safe = length > 1e-12
            corr = np.zeros_like(d)
            scale = np.zeros_like(length)
            scale[safe] = 0.5 * stiff * (length[safe] - rest) / length[safe]
            corr = d * scale[:, None]
            pos[i] += corr
            pos[j] -= corr

    def _contact_vertices(self, pos, prev):
        phi, normal = self._sdf(pos)
        pen = self.offset - phi
        hit = pen > 0.0
        if not np.any(hit):
            return
        idx = np.nonzero(hit)[0]
        nrm = normal[idx]
        depth = pen[idx][:, None]
        pos[idx] += nrm * depth
        if self.mu > 0.0:
            disp = pos[idx] - prev[idx]
            dn = np.sum(disp * nrm, axis=1)[:, None]
            tang = disp - dn * nrm
            tlen = np.sqrt(np.sum(tang * tang, axis=1))[:, None]
            limit = self.mu * depth
            # remove the whole tangential slip when it stays inside the
            # Coulomb cone (static), otherwise remove exactly the cone limit
            remove = np.where(tlen > limit,
                              limit / np.maximum(tlen, 1e-12),
                              1.0)
            pos[idx] -= tang * remove
    def _contact_aux_group(self, pos, groups):
        """Resolve contact at interior sample points (edge mids, quad centres).

        Prevents the sharp point / sharp edge from slipping between vertices.
        Corrections are redistributed equally to the supporting vertices.
        """
        k = groups.shape[1]
        sample = pos[groups].mean(axis=1)
        phi, normal = self._sdf(sample)
        pen = self.offset - phi
        hit = pen > 0.0
        if not np.any(hit):
            return
        sel = groups[hit]
        corr = normal[hit] * pen[hit][:, None]
        flat = sel.ravel()
        acc = np.zeros_like(pos)
        cnt = np.zeros(pos.shape[0])
        np.add.at(acc, flat, np.repeat(corr, k, axis=0))
        np.add.at(cnt, flat, 1.0)
        active = cnt > 0.0
        pos[active] += acc[active] / cnt[active, None]

    def _contact_self(self, pos):
        if _KDTree is None:
            return
        tree = _KDTree(pos)
        pairs = tree.query_pairs(self.self_radius, output_type="ndarray")
        if pairs.shape[0] == 0:
            return
        a = pairs[:, 0]
        b = pairs[:, 1]
        # ignore topological neighbours: their spacing is set by the
        # structural / shear constraints, not by cloth thickness
        far = (np.abs(self.grid_ix[a] - self.grid_ix[b]) > 2) | \
              (np.abs(self.grid_iy[a] - self.grid_iy[b]) > 2)
        if not np.any(far):
            return
        a = a[far]
        b = b[far]
        order = np.lexsort((b, a))
        a = a[order]
        b = b[order]
        d = pos[b] - pos[a]
        dist = np.sqrt(np.sum(d * d, axis=1))
        good = dist > 1e-9
        if not np.any(good):
            return
        a = a[good]
        b = b[good]
        d = d[good]
        dist = dist[good]
        pen = self.self_radius - dist
        push = 0.5 * (pen / dist)[:, None] * d
        acc = np.zeros_like(pos)
        cnt = np.zeros(pos.shape[0])
        np.add.at(acc, a, -push)
        np.add.at(acc, b, push)
        np.add.at(cnt, a, 1.0)
        np.add.at(cnt, b, 1.0)
        active = cnt > 0.0
        pos[active] += acc[active] / cnt[active, None]
    # -- time integration -------------------------------------------------
    def _substep(self, dt):
        prev = self.pos.copy()
        prev_vel = self.vel.copy()

        # symplectic-Euler prediction with linear aerodynamic drag
        damp = math.exp(-self.DAMPING_RATE * dt)
        vel = self.vel * damp + self.gravity[None, :] * dt
        speed = np.sqrt(np.sum(vel * vel, axis=1))
        over = speed > self.MAX_SPEED
        if np.any(over):
            vel[over] *= (self.MAX_SPEED / speed[over])[:, None]
        pos = prev + vel * dt

        # constraint + contact projection (Gauss-Seidel over colourings)
        for _ in range(self.ITERS):
            self._project_constraints(pos)
            self._contact_vertices(pos, prev)
        self._contact_aux_group(pos, self.aux_edges)
        self._contact_aux_group(pos, self.aux_quads)
        for _ in range(self.SELF_ITERS):
            self._contact_self(pos)
        self._contact_vertices(pos, prev)

        if not np.isfinite(pos).all():
            pos = np.where(np.isfinite(pos), pos, prev)
        np.clip(pos, -50.0, 50.0, out=pos)

        new_vel = (pos - prev) / dt
        if not np.isfinite(new_vel).all():
            new_vel = np.where(np.isfinite(new_vel), new_vel, prev_vel)
        speed = np.sqrt(np.sum(new_vel * new_vel, axis=1))
        over = speed > self.MAX_SPEED
        if np.any(over):
            new_vel[over] *= (self.MAX_SPEED / speed[over])[:, None]

        self.pos = pos
        self.vel = new_vel
        self._nsub += 1
        self._max_speed = max(self._max_speed, float(speed.max()))

    def _deformation_gradient(self):
        """Per-vertex deformation gradient from the discrete surface map."""
        nx, ny = self.nx, self.ny
        grid = self.pos.reshape(ny, nx, 3)
        du = np.gradient(grid, self.hx, axis=1)   # d x / d material-u
        dv = np.gradient(grid, self.hy, axis=0)   # d x / d material-v
        nrm = np.cross(du, dv)
        nlen = np.sqrt(np.sum(nrm * nrm, axis=-1))
        nrm = np.where(nlen[..., None] > 1e-12, nrm / np.maximum(nlen[..., None], 1e-12),
                       np.array([0.0, 0.0, 1.0]))
        F = np.stack([du.reshape(-1, 3), dv.reshape(-1, 3), nrm.reshape(-1, 3)], axis=2)
        return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t + 1e-12:
            span = target - self.t
            nsub = max(1, int(round(span / self.dt_sub)))
            dt = span / nsub
            for _ in range(nsub):
                self._substep(dt)
            self.t = target
        return self._make_frame(max(target, 0.0))

    def _make_frame(self, time: float) -> StateFrame:
        phi, _ = self._sdf(self.pos)
        gap = phi - self.offset
        gap = np.nan_to_num(gap, nan=0.0, posinf=1e3, neginf=-1e3)
        self._min_gap = min(self._min_gap, float(gap.min()))
        pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        frame = StateFrame(
            float(time),
            pos.copy(),
            vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": gap.astype(float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "substepped position-based cloth (self-implemented)",
            "substeps_taken": int(self._nsub),
            "substep_dt": float(self.dt_sub),
            "constraint_batches": int(len(self.batches)),
            "constraint_iterations": int(self.ITERS),
            "min_contact_gap": float(self._min_gap),
            "max_vertex_speed": float(self._max_speed),
            "friction_coefficient": float(self.mu),
            "contact_offset": float(self.offset),
            "self_contact": bool(_KDTree is not None),
        }
