"""Physical solver for scene 032_tablecloth_pull_2_mps.

Custom cloth simulation implemented directly with NumPy (no external
physics engine).  The physical model is:

* Cloth: the frozen 17x13 mass grid is integrated with XPBD distance
  constraints (structural / shear / bending) plus a hard stretch
  (strain) limiter on the structural edges, so the cloth is
  effectively inextensible while still being able to wrinkle.
* Pull: the -x cloth edge is a Dirichlet boundary condition following
  the prescribed horizontal pull x(t) = x0 - v * min(t, T) with
  v = 2 m/s and T = 0.9 s.  Everything else stays fully dynamic.
* Rigid dinnerware: translational rigid-body dynamics (one state
  sample per body) under gravity with frictional contact.
* Contact: signed-distance collision against the static table box and
  against the cylinder / box dinnerware primitives.  Penetrations are
  resolved with position projections; Coulomb friction is applied at
  the position level with a per-substep friction budget
  mu * (normal correction), which reproduces the physical sliding
  acceleration a = mu * g of the dishes while the cloth is yanked out
  from under them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - contract present at runtime
    StateFrame = object


# ----------------------------------------------------------------------
# signed distances and outward normals
# ----------------------------------------------------------------------
def _box_sdf(q, he):
    """SDF + outward unit normal of points q (relative to a box center)."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    he = np.asarray(he, dtype=float)
    cl = np.clip(q, -he, he)
    diff = q - cl
    dist = np.sqrt((diff * diff).sum(axis=1))
    outside = dist > 1e-12
    face = he - np.abs(q)
    sdf = np.where(outside, dist, -face.min(axis=1))
    n = np.zeros_like(q)
    out_idx = np.where(outside)[0]
    if out_idx.size:
        n[out_idx] = diff[out_idx] / dist[out_idx][:, None]
    in_idx = np.where(~outside)[0]
    if in_idx.size:
        axis = np.argmin(face[in_idx], axis=1)
        for k in range(3):
            m = axis == k
            if m.any():
                n[in_idx[m], k] = np.sign(q[in_idx[m], k])
    return sdf, n


def _cylinder_sdf(q, radius, half_height):
    """SDF + outward unit normal of points q relative to a z-axis cylinder."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    rad = np.hypot(q[:, 0], q[:, 1])
    az = np.abs(q[:, 2])
    dr = rad - radius
    dz = az - half_height
    outside = (dr > 0) | (dz > 0)
    sdf = np.where(
        outside,
        np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2),
        np.maximum(dr, dz),
    )
    n = np.zeros_like(q)
    out_idx = np.where(outside)[0]
    if out_idx.size:
        scale = np.where(rad[out_idx] > radius, radius / np.maximum(rad[out_idx], 1e-12), 1.0)
        cp = np.empty((out_idx.size, 3))
        cp[:, 0] = q[out_idx, 0] * scale
        cp[:, 1] = q[out_idx, 1] * scale
        cp[:, 2] = np.clip(q[out_idx, 2], -half_height, half_height)
        vec = q[out_idx] - cp
        ln = np.sqrt((vec * vec).sum(axis=1))
        ok = ln > 1e-12
        n[out_idx[ok]] = vec[ok] / ln[ok][:, None]
    in_idx = np.where(~outside)[0]
    if in_idx.size:
        radial_closer = dr[in_idx] >= dz[in_idx]
        m = radial_closer & (rad[in_idx] > 1e-12)
        n[in_idx[m], 0] = q[in_idx[m], 0] / rad[in_idx[m]]
        n[in_idx[m], 1] = q[in_idx[m], 1] / rad[in_idx[m]]
        m2 = ~radial_closer
        n[in_idx[m2], 2] = np.sign(q[in_idx[m2], 2])
    return sdf, n


class _Prim:
    """Collision primitive for one rigid dinnerware body."""

    def __init__(self, gidx, kind, mass, radius=0.0, height=0.0, half_extents=None):
        self.gidx = int(gidx)
        self.kind = kind
        self.mass = float(mass)
        self.radius = float(radius)
        self.half_height = 0.5 * float(height)
        self.he = None if half_extents is None else np.asarray(half_extents, dtype=float)
        self.half_z = self.half_height if kind == "cylinder" else float(self.he[2])
        self.w = 1.0 / self.mass

    def sdf(self, q):
        if self.kind == "cylinder":
            return _cylinder_sdf(q, self.radius, self.half_height)
        return _box_sdf(q, self.he)


# ----------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------
class BenchmarkSolver:
    """XPBD cloth + position-based frictional contact solver."""

    def __init__(self) -> None:
        self.t = 0.0
        self._ready = False

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config) if hasattr(scene, "config") else {}
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.asarray(frame0.body_id, dtype=int)
        self.orientation = np.array(frame0.orientation, dtype=float)
        self.angvel = np.zeros_like(self.v)
        self.uv = np.asarray(frame0.attributes.get("uv", np.zeros((self.n, 2))), dtype=float)
        self.mass_quantity = np.asarray(
            frame0.attributes.get("mass_quantity", self.mass), dtype=float
        )
        scal0 = dict(frame0.scalars)
        self.constraint_scale = float(scal0.get("constraint_scale", 1.9))
        self.contact_length_scale = float(scal0.get("contact_length_scale", 0.008))

        self.n = len(self.x)
        self.g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.pull_speed = float(cfg.get("pull_speed", 2.0))
        self.pull_duration = float(cfg.get("pull_duration", 0.9))

        cloth_cfg = dict(cfg.get("cloth", {}))
        grid = cloth_cfg.get("grid_shape") or [17, 13]
        self.nx, self.ny = int(grid[0]), int(grid[1])
        self.thickness = float(cloth_cfg.get("thickness", 0.008))
        self.r_node = 0.5 * self.thickness

        # cloth / rigid partition from the frozen state binding
        self.cloth_idx = np.where(self.body_id == 0)[0]
        if self.uv.shape != (self.n, 2):
            self.uv = np.zeros((self.n, 2), dtype=float)
        if self.mass_quantity.shape != (self.n,):
            self.mass_quantity = self.mass.copy()

        # frozen scene geometry
        geo = cfg.get("scene_geometry")
        if geo is None:
            geo = self._load_geometry_yaml()
        self.table_c = np.zeros(3)
        self.table_he = np.array([1.25, 0.9, 0.06])
        self.prim_list: list[_Prim] = []
        if geo:
            for obj in geo.get("objects", []):
                geom = obj.get("geometry", {})
                tr = np.asarray(obj.get("transform", {}).get("translation", [0, 0, 0]), dtype=float)
                binding = obj.get("state_binding") or {}
                bid = binding.get("value") if binding.get("selector") == "body_id" else None
                semantic = obj.get("semantic", "")
                if semantic == "boundary" or obj.get("id") == "table":
                    self.table_c = tr
                    self.table_he = np.asarray(geom.get("half_extents", [1.25, 0.9, 0.06]), dtype=float)
                elif semantic == "rigid" and bid is not None:
                    hits = np.where(self.body_id == int(bid))[0]
                    if hits.size:
                        self.prim_list.append(
                            _Prim(
                                hits[0],
                                geom.get("type", "box"),
                                float(self.mass[hits[0]]),
                                radius=float(geom.get("radius", 0.0) or 0.0),
                                height=float(geom.get("height", 0.0) or 0.0),
                                half_extents=geom.get("half_extents"),
                            )
                        )
        self.table_top = float(self.table_c[2] + self.table_he[2])

        # cloth inverse masses
        self._inv_mass_cloth = 1.0 / np.maximum(self.mass[self.cloth_idx], 1e-12)
        self._w_full = np.zeros(self.n)
        self._w_full[self.cloth_idx] = self._inv_mass_cloth

        # pulled edge: i == 0 column of the grid, pulled toward -x
        self.edge_local = np.array([j * self.nx for j in range(self.ny)], dtype=int)
        self.edge_gidx = self.cloth_idx[self.edge_local]
        self.edge_x0 = self.x[self.edge_gidx, 0].copy()
        self.edge_x0_mean = float(self.edge_x0.mean())
        self.edge_y0_mean = float(self.x[self.edge_gidx, 1].mean())

        # distance constraints (global indices)
        ia: list[int] = []
        ib: list[int] = []
        kind: list[int] = []
        x0 = self.x
        cloth = self.cloth_idx
        for j in range(self.ny):
            for i in range(self.nx):
                a = j * self.nx + i
                if i + 1 < self.nx:
                    ia.append(a); ib.append(a + 1); kind.append(0)
                if j + 1 < self.ny:
                    ia.append(a); ib.append(a + self.nx); kind.append(0)
                if i + 1 < self.nx and j + 1 < self.ny:
                    ia.append(a); ib.append(a + self.nx + 1); kind.append(1)
                    ia.append(a + 1); ib.append(a + self.nx); kind.append(1)
                if i + 2 < self.nx:
                    ia.append(a); ib.append(a + 2); kind.append(2)
                if j + 2 < self.ny:
                    ia.append(a); ib.append(a + 2 * self.nx); kind.append(2)
        self.ia = cloth[np.asarray(ia, dtype=int)]
        self.ib = cloth[np.asarray(ib, dtype=int)]
        kinds = np.asarray(kind, dtype=int)
        self.rest = np.sqrt(((x0[self.ia] - x0[self.ib]) ** 2).sum(axis=1))
        # XPBD compliance: structural/shear nearly inextensible, bending soft
        alpha_map = {0: 1.0e-9, 1: 1.0e-8, 2: 3.0e-2}
        self.alpha = np.array([alpha_map[int(k)] for k in kinds], dtype=float)
        self.struct_mask = kinds == 0
        self.ia_s = self.ia[self.struct_mask]
        self.ib_s = self.ib[self.struct_mask]
        self.rest_s = self.rest[self.struct_mask]

        # numerical parameters
        self.h_max = 0.01          # paper time step
        self.iters = 20            # XPBD sweeps per substep
        self.relax = 0.5           # Jacobi relaxation
        self.strain_passes = 4
        self.strain_relax = 0.8
        self.strain_tol = 0.02     # 2% stretch limit on structural edges
        self.rounds = 4            # contact iterations per substep
        self.mu_ct = 0.30          # cloth-table friction
        self.mu_cr = 0.55          # cloth-dinnerware friction
        self.mu_rt = 0.45          # dinnerware-table friction
        self.damping = 0.2         # mild linear drag [1/s]
        self.v_max = 30.0

        nb = len(self.prim_list)
        self._C_cloth = np.zeros(nb)
        self._C_table = np.zeros(nb)
        self._node_disp = np.zeros((nb, 3))
        self._pull_j = np.zeros((len(self.edge_gidx), 3))
        self._frame_dt = 0.0
        self._nsub = 0

        self.t = 0.0
        self._ready = True

    @staticmethod
    def _load_geometry_yaml():
        try:
            path = Path(__file__).resolve().parents[1] / "scene_geometry.yaml"
            if path.is_file():
                from worldsolver.config import load_yaml

                return load_yaml(path)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t = float(time)
        if t < 0.0:
            t = 0.0
        if t < self.t:
            t = self.t
        rem = t - self.t
        self._pull_j = np.zeros((len(self.edge_gidx), 3))
        self._frame_dt = rem
        if rem > 1e-12:
            nsub = max(1, int(math.ceil(rem / self.h_max - 1e-9)))
            dt = rem / nsub
            for _ in range(nsub):
                self._step(dt)
        return self._build_frame(t)

    def _edge_x(self, t: float) -> np.ndarray:
        s = min(t, self.pull_duration)
        return self.edge_x0 - self.pull_speed * s

    def _step(self, dt: float) -> None:
        t_old = self.t
        t_new = t_old + dt
        pulling = t_new <= self.pull_duration + 1e-12

        x_prev = self.x.copy()

        # gravity + mild drag, explicit Euler prediction
        self.v += self.g[None, :] * dt
        self.v *= max(0.0, 1.0 - self.damping * dt)
        x = self.x + self.v * dt

        # cloth constraints (edge pinned in x while the pull is active)
        w = self._w_full.copy()
        if pulling:
            w[self.edge_gidx] = 0.0
        self._solve_distance(x, dt, w)
        self._strain_limit(x, w)
        x_mid = x.copy()  # state before contact resolution

        # contact rounds
        for _ in range(self.rounds):
            if pulling:
                x[self.edge_gidx, 0] = self._edge_x(t_new)
            self._C_cloth[:] = 0.0
            self._C_table[:] = 0.0
            self._node_disp[:] = 0.0
            self._collide_table_cloth(x, x_prev)
            self._collide_cloth_rigid(x, x_prev)
            self._collide_rigid_table(x)
            self._rigid_friction(x, x_prev)

        if pulling:
            x[self.edge_gidx, 0] = self._edge_x(t_new)
            # bookkeeping of the pull (boundary) force: impulse needed to
            # enforce the prescribed edge motion on top of the free motion
            v_free = (x_mid[self.edge_gidx] - x_prev[self.edge_gidx]) / dt
            v_bc = (self._edge_x(t_new) - self._edge_x(t_old)) / dt
            self._pull_j[:, 0] += self.mass[self.edge_gidx] * (v_bc - v_free[:, 0])

        # velocities from finite differences of the projected positions
        v = (x - x_prev) / dt
        speed = np.sqrt((v * v).sum(axis=1))
        scale = np.minimum(1.0, self.v_max / np.maximum(speed, 1e-12))
        v *= scale[:, None]
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        self.x = x
        self.v = v
        self.t = t_new
        self._nsub += 1

    # ------------------------------------------------------------------
    def _solve_distance(self, x: np.ndarray, dt: float, w: np.ndarray) -> None:
        ia, ib = self.ia, self.ib
        wsum = w[ia] + w[ib]
        alpha_t = self.alpha / (dt * dt)
        lam = np.zeros(len(ia))
        for _ in range(self.iters):
            d = x[ib] - x[ia]
            leng = np.sqrt((d * d).sum(axis=1))
            ok = leng > 1e-12
            c = leng - self.rest
            denom = np.maximum(wsum + alpha_t, 1e-12)
            dlam = np.where(ok, (-c - alpha_t * lam) / denom, 0.0)
            lam = lam + dlam
            nvec = d / np.maximum(leng, 1e-12)[:, None]
            dx = np.zeros_like(x)
            np.add.at(dx, ia, (-dlam * w[ia])[:, None] * nvec)
            np.add.at(dx, ib, (dlam * w[ib])[:, None] * nvec)
            x += self.relax * dx

    def _strain_limit(self, x: np.ndarray, w: np.ndarray) -> None:
        ia, ib = self.ia_s, self.ib_s
        hi = self.rest_s * (1.0 + self.strain_tol)
        wsum = w[ia] + w[ib]
        for _ in range(self.strain_passes):
            d = x[ib] - x[ia]
            leng = np.sqrt((d * d).sum(axis=1))
            c = leng - hi
            act = (c > 0) & (leng > 1e-12)
            dlam = np.where(act, -c / np.maximum(wsum, 1e-12), 0.0)
            nvec = d / np.maximum(leng, 1e-12)[:, None]
            dx = np.zeros_like(x)
            np.add.at(dx, ia, (-dlam * w[ia])[:, None] * nvec)
            np.add.at(dx, ib, (dlam * w[ib])[:, None] * nvec)
            x += self.strain_relax * dx

    # ------------------------------------------------------------------
    # contact
    # ------------------------------------------------------------------
    def _supported(self, p: np.ndarray) -> np.ndarray:
        """Cloth nodes resting on the table (transmit load to the world)."""
        return (
            (p[:, 2] < self.table_top + self.r_node + 0.02)
            & (np.abs(p[:, 0] - self.table_c[0]) < self.table_he[0])
            & (np.abs(p[:, 1] - self.table_c[1]) < self.table_he[1])
        )

    def _collide_table_cloth(self, x: np.ndarray, x_prev: np.ndarray) -> None:
        ci = self.cloth_idx
        q = x[ci] - self.table_c
        sdf, n = _box_sdf(q, self.table_he)
        gap = sdf - self.r_node
        pen = gap < -1e-12
        if not pen.any():
            return
        idx = np.where(pen)[0]
        push = -gap[idx]
        nb = n[idx]
        x[ci[idx]] += nb * push[:, None]
        # Coulomb friction vs the static table: budget mu * normal correction
        disp = x[ci[idx]] - x_prev[ci[idx]]
        disp_n = (disp * nb).sum(axis=1)
        disp_t = disp - nb * disp_n[:, None]
        lt = np.sqrt((disp_t * disp_t).sum(axis=1))
        lim = self.mu_ct * push
        corr = np.minimum(lim, lt)
        shrink = np.where(lt > 1e-15, corr / np.maximum(lt, 1e-15), 0.0)
        x[ci[idx]] -= disp_t * shrink[:, None]

    def _collide_cloth_rigid(self, x: np.ndarray, x_prev: np.ndarray) -> None:
        ci = self.cloth_idx
        p = x[ci]
        pprev = x_prev[ci]
        for b, prim in enumerate(self.prim_list):
            gi = prim.gidx
            q = p - x[gi]
            sdf, n = prim.sdf(q)
            gap = sdf - self.r_node
            pen = gap < -1e-12
            if not pen.any():
                continue
            idx = np.where(pen)[0]
            c = -gap[idx]
            nb = n[idx]
            sup = (nb[:, 2] < -0.5) & self._supported(p[idx])

            # unsupported contacts: mass-weighted split
            u = np.where(~sup)[0]
            if u.size:
                wnu = self._inv_mass_cloth[idx[u]]
                wr = prim.w
                fru = wr / np.maximum(wnu + wr, 1e-12)
                fnu = wnu / np.maximum(wnu + wr, 1e-12)
                x[ci[idx[u]]] += nb[u] * (c[u] * fnu)[:, None]
                x[gi] -= (nb[u] * (c[u] * fru)[:, None]).sum(axis=0)

            # supported contacts: the table braces the cloth node, so the
            # rigid body receives the full separation correction
            s = np.where(sup)[0]
            if s.size:
                cs = c[s]
                ns = nb[s]
                nbar = (ns * cs[:, None]).sum(axis=0)
                nn = float(np.linalg.norm(nbar))
                if nn > 1e-12:
                    nbar = nbar / nn
                cmax = float(cs.max())
                x[gi] -= nbar * cmax
                self._C_cloth[b] += cmax
                self._node_disp[b] = (p[idx[s]] - pprev[idx[s]]).mean(axis=0)

    def _collide_rigid_table(self, x: np.ndarray) -> None:
        for b, prim in enumerate(self.prim_list):
            gi = prim.gidx
            bottom = x[gi].copy()
            bottom[2] -= prim.half_z
            sdf, n = _box_sdf((bottom - self.table_c)[None, :], self.table_he)
            if sdf[0] < -1e-12:
                c = -float(sdf[0])
                x[gi] += n[0] * c
                self._C_table[b] += c

    def _rigid_friction(self, x: np.ndarray, x_prev: np.ndarray) -> None:
        for b, prim in enumerate(self.prim_list):
            limit = self.mu_cr * self._C_cloth[b] + self.mu_rt * self._C_table[b]
            if limit <= 1e-15:
                continue
            gi = prim.gidx
            rel = (x[gi] - x_prev[gi]) - self._node_disp[b]
            rel = np.array([rel[0], rel[1], 0.0])
            m = float(np.linalg.norm(rel))
            if m > 1e-15:
                corr = min(limit, m)
                x[gi] -= rel * (corr / m)

    # ------------------------------------------------------------------
    # frame export
    # ------------------------------------------------------------------
    def _build_frame(self, t: float):
        csd = np.zeros(self.n)
        ci = self.cloth_idx
        p = self.x[ci]
        d_tab, _ = _box_sdf(p - self.table_c, self.table_he)
        best = d_tab - self.r_node
        for prim in self.prim_list:
            d2, _ = prim.sdf(p - self.x[prim.gidx])
            best = np.minimum(best, d2 - self.r_node)
        csd[ci] = best
        for prim in self.prim_list:
            d2, _ = prim.sdf(p - self.x[prim.gidx])
            gap_cloth = float((d2 - self.r_node).min())
            bottom = self.x[prim.gidx].copy()
            bottom[2] -= prim.half_z
            d_tab, _ = _box_sdf((bottom - self.table_c)[None, :], self.table_he)
            csd[prim.gidx] = min(gap_cloth, float(d_tab[0]))

        # constraint: horizontal displacement of the pulled cloth edge
        value = np.array(
            [
                float(self.x[self.edge_gidx, 0].mean()) - self.edge_x0_mean,
                float(self.x[self.edge_gidx, 1].mean()) - self.edge_y0_mean,
            ]
        )
        s = min(max(float(t), 0.0), self.pull_duration)
        target = np.array([-self.pull_speed * s, 0.0])

        # external force: gravity + prescribed pull reaction on the edge
        fext = self.mass[:, None] * self.g[None, :]
        if self._frame_dt > 1e-12 and np.any(self._pull_j):
            fext[self.edge_gidx, 0] += self._pull_j[:, 0] / self._frame_dt

        frame = StateFrame(
            float(t),
            self.x.copy(),
            self.v.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angvel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "contact_signed_distance": csd,
                "constraint_value": value,
                "constraint_target": target,
                "cumulative_mass_source": np.zeros(1),
                "external_force": fext,
            },
            scalars={
                "constraint_scale": self.constraint_scale,
                "contact_length_scale": self.contact_length_scale,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        on_table = 0
        names = []
        geo = self._load_geometry_yaml() or {}
        name_by_body = {}
        for obj in geo.get("objects", []):
            binding = obj.get("state_binding") or {}
            if binding.get("selector") == "body_id":
                name_by_body[int(binding["value"])] = obj.get("id", "?")
        for prim in self.prim_list:
            pos = self.x[prim.gidx]
            bottom = float(pos[2]) - prim.half_z
            inside = (
                abs(float(pos[0]) - self.table_c[0]) < self.table_he[0]
                and abs(float(pos[1]) - self.table_c[1]) < self.table_he[1]
            )
            if inside and (self.table_top - 0.05) <= bottom <= (self.table_top + 0.10):
                on_table += 1
                names.append(name_by_body.get(self.body_id[prim.gidx], "?"))
        return {
            "method": "xpbd-cloth + position-based frictional contact (numpy)",
            "substeps": int(self._nsub),
            "final_time": float(self.t),
            "objects_on_table": int(on_table),
            "objects_on_table_names": sorted(names),
        }
