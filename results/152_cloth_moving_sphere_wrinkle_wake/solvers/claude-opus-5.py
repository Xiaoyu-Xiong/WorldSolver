"""Mass-spring thin-shell cloth solver: wrinkle wake from a sphere in contact.

Physical model
--------------
The cloth is a lumped-mass discretization of the frozen 17x17 grid (289 nodes,
exact topology order of ``assets/cloth_topology.json``).  Internal elasticity is
a three-family spring network on the material grid:

* structural springs along the material x and y directions (near-inextensible
  membrane response),
* diagonal shear springs on every quad (in-plane shear),
* flexion springs across two cells (weak bending, so compression buckles into
  short-wavelength wrinkles instead of shortening the sheet).

Each family carries axial Rayleigh damping.  Time stepping is explicit
symplectic (semi-implicit) Euler with a substep well below the stiffest
critical step, followed by non-penetration projection against the frozen
rigid sphere and ground box with Coulomb friction, and prescribed kinematic
rails on the two material edges x = +/- Lx/2.

The sphere is a frozen world-fixed obstacle, so the prescribed relative sweep
between sphere and cloth is imposed by translating the cloth rails.  In the
sphere frame this is exactly the paper's setup: a rigid sphere travelling
against a gently draped sheet, producing a travelling contact bulge and an
anisotropic wrinkle wake trailing through the fabric.

Everything is deterministic: fixed substep, fixed seeded initial perturbation,
no adaptivity that depends on wall-clock time.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]

        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.sx, self.sy = (float(v) for v in cloth["size"])
        self.center = np.asarray(cloth["center"], dtype=float)
        self.thickness = float(cloth["thickness"])
        self.total_mass = float(cloth["total_mass"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.n_nodes = self.nx * self.ny

        self.hx = self.sx / (self.nx - 1)
        self.hy = self.sy / (self.ny - 1)

        # ---- frozen obstacle geometry (from scene_geometry.yaml) ----------
        self.sphere_center = np.array([0.0, 0.0, 0.68])
        self.sphere_radius = 0.62
        self.ground_top = 0.0
        geometry = cfg.get("scene_geometry") or {}
        for item in geometry.get("objects", []) or []:
            geo = item.get("geometry", {}) or {}
            tr = (item.get("transform", {}) or {}).get("translation", None)
            if item.get("id") == "sphere" and geo.get("type") == "sphere":
                self.sphere_radius = float(geo["radius"])
                if tr is not None:
                    self.sphere_center = np.asarray(tr, dtype=float)
            elif item.get("id") == "ground" and geo.get("type") == "box":
                half = np.asarray(geo["half_extents"], dtype=float)
                base = np.asarray(tr, dtype=float) if tr is not None else np.zeros(3)
                self.ground_top = float(base[2] + half[2])

        self.half_thickness = 0.5 * self.thickness
        self.contact_radius = self.sphere_radius + self.half_thickness
        self.ground_level = self.ground_top + self.half_thickness

        # ---- material / numerical parameters -----------------------------
        self.mass_node = self.total_mass / self.n_nodes
        self.mass = np.full(self.n_nodes, self.mass_node)
        self.mass_col = self.mass[:, None]

        self.k_struct = 600.0        # N/m  membrane
        self.k_shear = 120.0         # N/m  in-plane shear
        self.k_flex = 4.0            # N/m  weak bending -> wrinkles
        self.c_struct = 0.040        # N s/m axial damping
        self.c_shear = 0.020
        self.c_flex = 0.004
        self.air_drag = 0.30         # 1/s  aerodynamic drag coefficient
        self.mu_sphere = 0.35        # Coulomb friction, cloth/sphere
        self.mu_ground = 0.50

        self.dt_max = 1.0e-4         # ~12x below the explicit stability limit

        # ---- prescribed relative sweep -----------------------------------
        self.slack = 0.025           # in-plane slack -> gentle drape
        self.x_offset = -0.50        # start upstream of the sphere
        self.sweep_speed = 0.20      # m/s cruise speed of the relative motion
        self.sweep_ramp = 1.0        # s  smooth ease-in

        # Counters must exist before the pre-roll settle loop touches them.
        self.time = 0.0
        self.ext_impulse = np.zeros((self.n_nodes, 3))
        self.interval = 0.0
        self.n_substeps = 0
        self.max_penetration = 0.0
        self.max_speed = 0.0

        self._build_topology()
        self._build_initial_state(int(seed))

        self.time = 0.0
        self.interval = 0.0
        self.ext_impulse = np.zeros((self.n_nodes, 3))

    # ---------------------------------------------------------- topology
    def _index(self, i, j):
        """Node ordering used by scene.py and cloth_topology.json (x fastest)."""
        return j * self.nx + i

    def _build_topology(self):
        nx, ny = self.nx, self.ny
        i_idx = []
        j_idx = []
        rest = []
        stiff = []
        damp = []

        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")

        def add(a, b, length, k, c):
            a = np.asarray(a).ravel()
            b = np.asarray(b).ravel()
            i_idx.append(a)
            j_idx.append(b)
            rest.append(np.full(a.size, float(length)))
            stiff.append(np.full(a.size, float(k)))
            damp.append(np.full(a.size, float(c)))

        # structural springs
        a = self._index(ii[:, :-1], jj[:, :-1])
        b = self._index(ii[:, :-1] + 1, jj[:, :-1])
        add(a, b, self.hx, self.k_struct, self.c_struct)

        a = self._index(ii[:-1, :], jj[:-1, :])
        b = self._index(ii[:-1, :], jj[:-1, :] + 1)
        add(a, b, self.hy, self.k_struct, self.c_struct)

        # shear diagonals on every quad
        diag = float(np.hypot(self.hx, self.hy))
        a = self._index(ii[:-1, :-1], jj[:-1, :-1])
        b = self._index(ii[:-1, :-1] + 1, jj[:-1, :-1] + 1)
        add(a, b, diag, self.k_shear, self.c_shear)

        a = self._index(ii[:-1, :-1] + 1, jj[:-1, :-1])
        b = self._index(ii[:-1, :-1], jj[:-1, :-1] + 1)
        add(a, b, diag, self.k_shear, self.c_shear)

        # flexion (bending) springs spanning two cells
        a = self._index(ii[:, :-2], jj[:, :-2])
        b = self._index(ii[:, :-2] + 2, jj[:, :-2])
        add(a, b, 2.0 * self.hx, self.k_flex, self.c_flex)

        a = self._index(ii[:-2, :], jj[:-2, :])
        b = self._index(ii[:-2, :], jj[:-2, :] + 2)
        add(a, b, 2.0 * self.hy, self.k_flex, self.c_flex)

        self.sI = np.concatenate(i_idx).astype(np.int64)
        self.sJ = np.concatenate(j_idx).astype(np.int64)
        self.sL0 = np.concatenate(rest)
        self.sk = np.concatenate(stiff)
        self.sc = np.concatenate(damp)
        n_springs = self.sI.size

        # signed incidence transpose (N x S) for a single sparse scatter
        cols = np.concatenate([np.arange(n_springs), np.arange(n_springs)])
        rows = np.concatenate([self.sI, self.sJ])
        vals = np.concatenate([np.ones(n_springs), -np.ones(n_springs)])
        self.inc_t = sp.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_nodes, n_springs)
        )

        # kinematic rails: the two material edges x = +/- Lx/2
        pin = (ii == 0) | (ii == nx - 1)
        self.pin_idx = np.flatnonzero(pin.ravel())
        self.boundary_vertex = (
            (ii == 0) | (ii == nx - 1) | (jj == 0) | (jj == ny - 1)
        ).ravel().astype(np.int32)

        # material coordinates (for the pin targets and deformation gradient)
        x = np.linspace(-0.5 * self.sx, 0.5 * self.sx, nx)
        y = np.linspace(-0.5 * self.sy, 0.5 * self.sy, ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        self.mat_x = xx.ravel()
        self.mat_y = yy.ravel()

    # ------------------------------------------------- prescribed motion
    def _sweep(self, t):
        """Smooth C1 sweep displacement s(t) and speed ds/dt (relative motion)."""
        if t <= 0.0:
            return 0.0, 0.0
        u0 = self.sweep_speed
        ramp = self.sweep_ramp
        if t < ramp:
            u = t / ramp
            speed = u0 * (3.0 * u * u - 2.0 * u * u * u)
            disp = u0 * ramp * (u**3 - 0.5 * u**4)
        else:
            speed = u0
            disp = u0 * (0.5 * ramp + (t - ramp))
        return float(disp), float(speed)

    def _pin_state(self, t):
        disp, speed = self._sweep(t)
        idx = self.pin_idx
        pos = np.empty((idx.size, 3))
        pos[:, 0] = self.mat_x[idx] * (1.0 - self.slack) + self.x_offset + disp
        pos[:, 1] = self.mat_y[idx]
        pos[:, 2] = self.center[2]
        vel = np.zeros((idx.size, 3))
        vel[:, 0] = speed
        return pos, vel

    # ------------------------------------------------------ initial state
    def _build_initial_state(self, seed):
        p = np.empty((self.n_nodes, 3))
        p[:, 0] = self.mat_x * (1.0 - self.slack) + self.x_offset
        p[:, 1] = self.mat_y
        p[:, 2] = self.center[2]

        # deterministic micro-perturbation so compression can pick a
        # buckling mode instead of resting on an unstable flat branch
        rng = np.random.default_rng(seed)
        p[:, 2] += 1.0e-4 * rng.standard_normal(self.n_nodes)

        self.p = p
        self.v = np.zeros_like(p)
        self._project_out_of_sphere_initial()

        # quasi-static settle: the sheet drapes gently onto the sphere before t=0
        # Pre-roll ends exactly at t = 0 so the sweep (which is identically zero
        # for t <= 0) is fully available to the recorded window.
        settle_hi, settle_lo = 0.7, 0.3
        self.time = -(settle_hi + settle_lo)
        self.ext_impulse = np.zeros((self.n_nodes, 3))
        for _ in range(int(round(settle_hi / self.dt_max))):
            self._substep(self.dt_max, extra_damping=4.0)
        for _ in range(int(round(settle_lo / self.dt_max))):
            self._substep(self.dt_max, extra_damping=0.8)
        self.time = 0.0
        self.v[self.pin_idx] = 0.0

    def _project_out_of_sphere_initial(self):
        r = self.p - self.sphere_center
        dist = np.linalg.norm(r, axis=1)
        inside = dist < self.contact_radius
        if np.any(inside):
            safe = np.maximum(dist[inside], 1.0e-9)
            n = r[inside] / safe[:, None]
            self.p[inside] = self.sphere_center + self.contact_radius * n

    # ------------------------------------------------------------ physics
    def _internal_forces(self, p, v):
        d = p[self.sI] - p[self.sJ]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        inv = 1.0 / np.maximum(length, 1.0e-12)
        u = d * inv[:, None]
        rel = np.einsum("ij,ij->i", v[self.sI] - v[self.sJ], u)
        f_mag = self.sk * (length - self.sL0) + self.sc * rel
        f_vec = f_mag[:, None] * u
        # node i pulled toward node j when stretched
        return self.inc_t.dot(-f_vec)

    def _substep(self, h, extra_damping=0.0):
        p, v = self.p, self.v

        f_int = self._internal_forces(p, v)
        f_ext = self.mass_col * self.gravity[None, :] - (
            self.air_drag * self.mass_col * v
        )
        self.ext_impulse += h * f_ext
        damp = extra_damping

        acc = (f_int + f_ext) / self.mass_col
        if damp > 0.0:
            acc -= damp * v
        v += h * acc
        p += h * v

        self._resolve_sphere()
        self._resolve_ground()
        self._apply_pins(self.time + h)

        self.time += h
        self.n_substeps += 1

    def _resolve_sphere(self):
        p, v = self.p, self.v
        r = p - self.sphere_center
        dist = np.sqrt(np.einsum("ij,ij->i", r, r))
        hit = dist < self.contact_radius
        if not np.any(hit):
            return
        pen = float(np.max(self.contact_radius - dist[hit]))
        if pen > self.max_penetration:
            self.max_penetration = pen
        safe = np.maximum(dist[hit], 1.0e-9)
        n = r[hit] / safe[:, None]
        p[hit] = self.sphere_center + self.contact_radius * n

        v_old = v[hit].copy()
        vn = np.einsum("ij,ij->i", v_old, n)
        vt = v_old - vn[:, None] * n
        vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        approaching = vn < 0.0
        scale = np.clip(
            1.0 - self.mu_sphere * np.abs(vn) / np.maximum(vt_mag, 1.0e-9), 0.0, 1.0
        )
        scale = np.where(approaching, scale, 1.0)
        vn_new = np.where(approaching, 0.0, vn)
        v_new = vt * scale[:, None] + vn_new[:, None] * n
        v[hit] = v_new
        self.ext_impulse[hit] += self.mass_col[hit] * (v_new - v_old)

    def _resolve_ground(self):
        p, v = self.p, self.v
        hit = p[:, 2] < self.ground_level
        if not np.any(hit):
            return
        p[hit, 2] = self.ground_level
        v_old = v[hit].copy()
        vz = v_old[:, 2]
        approaching = vz < 0.0
        vt = v_old.copy()
        vt[:, 2] = 0.0
        vt_mag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        scale = np.clip(
            1.0 - self.mu_ground * np.abs(vz) / np.maximum(vt_mag, 1.0e-9), 0.0, 1.0
        )
        scale = np.where(approaching, scale, 1.0)
        v_new = vt * scale[:, None]
        v_new[:, 2] = np.where(approaching, 0.0, vz)
        v[hit] = v_new
        self.ext_impulse[hit] += self.mass_col[hit] * (v_new - v_old)

    def _apply_pins(self, t):
        idx = self.pin_idx
        pos, vel = self._pin_state(t)
        v_old = self.v[idx].copy()
        self.p[idx] = pos
        self.v[idx] = vel
        self.ext_impulse[idx] += self.mass_col[idx] * (vel - v_old)

    # ------------------------------------------------------- diagnostics
    def _deformation_gradient(self):
        """Per-node in-plane deformation gradient in material coordinates."""
        pos = self.p.reshape(self.ny, self.nx, 3)
        d_dx = np.gradient(pos, self.hx, axis=1)
        d_dy = np.gradient(pos, self.hy, axis=0)
        j1 = d_dx.reshape(-1, 3)
        j2 = d_dy.reshape(-1, 3)

        n1 = np.sqrt(np.einsum("ij,ij->i", j1, j1))
        e1 = j1 / np.maximum(n1, 1.0e-12)[:, None]
        proj = np.einsum("ij,ij->i", j2, e1)
        w = j2 - proj[:, None] * e1
        n2 = np.sqrt(np.einsum("ij,ij->i", w, w))
        e2 = w / np.maximum(n2, 1.0e-12)[:, None]

        f = np.zeros((self.n_nodes, 2, 2))
        f[:, 0, 0] = n1
        f[:, 0, 1] = proj
        f[:, 1, 0] = np.einsum("ij,ij->i", j1, e2)
        f[:, 1, 1] = n2
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def _signed_distance(self):
        r = self.p - self.sphere_center
        d_sphere = np.sqrt(np.einsum("ij,ij->i", r, r)) - self.contact_radius
        d_ground = self.p[:, 2] - self.ground_level
        return np.minimum(d_sphere, d_ground)

    def _external_force(self):
        if self.interval > 0.0:
            return self.ext_impulse / self.interval
        return self.mass_col * self.gravity[None, :] - (
            self.air_drag * self.mass_col * self.v
        )

    # ---------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        self.ext_impulse = np.zeros((self.n_nodes, 3))
        self.interval = max(target - self.time, 0.0)
        if self.interval > 0.0:
            n = int(np.ceil(self.interval / self.dt_max))
            h = self.interval / n
            for _ in range(n):
                self._substep(h)
            self.time = target

        speed = float(np.max(np.linalg.norm(self.v, axis=1)))
        self.max_speed = max(self.max_speed, speed)

        position = np.ascontiguousarray(self.p, dtype=float)
        velocity = np.ascontiguousarray(self.v, dtype=float)
        if not (np.isfinite(position).all() and np.isfinite(velocity).all()):
            raise FloatingPointError("cloth state diverged")

        frame = StateFrame(
            time=target,
            position=position.copy(),
            velocity=velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n_nodes, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_vertex.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "external_force": self._external_force(),
                "contact_signed_distance": self._signed_distance(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "lumped mass-spring shell, symplectic Euler + projected contact",
            "substep": float(self.dt_max),
            "substeps_taken": int(self.n_substeps),
            "nodes": int(self.n_nodes),
            "springs": int(self.sI.size),
            "max_sphere_penetration_pre_projection": float(self.max_penetration),
            "max_node_speed": float(self.max_speed),
            "sweep_speed": float(self.sweep_speed),
            "total_mass": float(self.total_mass),
        }
