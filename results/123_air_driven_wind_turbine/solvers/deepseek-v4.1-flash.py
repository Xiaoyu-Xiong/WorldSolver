"""Air-driven three-blade wind turbine with a resolved helical wake.

Self-contained multiphysics solver:

* Incompressible Navier-Stokes on a staggered (MAC) uniform grid solved with
  semi-Lagrangian advection and a DST-I pressure projection (p = 0 on the
  open faces, no-slip/no-penetration on the ground, prescribed inlet).
* The three-blade rotor is a dynamic rigid body on a fixed revolute axle.
  Each blade is discretised into lifting blade elements built from the frozen
  mesh (span, taper, chord, mass, inertia).  The local relative wind produces
  the aerodynamic force; the equal and opposite force is scattered back onto
  the fluid (actuator line), so the coupling is two-way.
* Axle friction is the regularised viscous + Coulomb resisting torque of the
  frozen configuration; the rotor is never driven by a motor.
* Passive coloured smoke is emitted at the rotating blade tips and advected by
  the simulated velocity field with a second order limited scheme, revealing
  the helical wake.

Everything is deterministic and uses only NumPy / SciPy.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import fft as sp_fft

from worldsolver.contracts import StateFrame

R_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _smoothstep(x: float) -> float:
    s = min(max(float(x), 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _quat_x(angle: float) -> np.ndarray:
    """xyzw quaternion for a rotation about the world x axis."""
    half = 0.5 * angle
    return np.array([math.sin(half), 0.0, 0.0, math.cos(half)], dtype=float)


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg

        dom = cfg["domain"]
        self.bmin = np.asarray(dom["bounds_min"], dtype=float)
        self.bmax = np.asarray(dom["bounds_max"], dtype=float)
        self.nx, self.ny, self.nz = (int(v) for v in dom["reference_grid"])
        self.shape = (self.nx, self.ny, self.nz)
        extent = self.bmax - self.bmin
        self.h = float(extent[0] / self.nx)
        self.rho = float(cfg["air"]["density"])
        self.nu = 1.5e-5

        # frozen grid coordinates -------------------------------------------------
        self.xf = self.bmin[0] + self.h * np.arange(self.nx + 1)
        self.yf = self.bmin[1] + self.h * np.arange(self.ny + 1)
        self.zf = self.bmin[2] + self.h * np.arange(self.nz + 1)
        self.xc = self.bmin[0] + self.h * (np.arange(self.nx) + 0.5)
        self.yc = self.bmin[1] + self.h * (np.arange(self.ny) + 0.5)
        self.zc = self.bmin[2] + self.h * (np.arange(self.nz) + 0.5)
        ZX, ZY, ZZ = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")
        self.CX, self.CY, self.CZ = ZX, ZY, ZZ
        # outward unit normal per open boundary / wall
        self.turbine = cfg["turbine"]
        self.hub = np.asarray(self.turbine["hub_center"], dtype=float)
        self.axle = np.asarray(self.turbine["axle_axis"], dtype=float)
        self.axle = self.axle / np.linalg.norm(self.axle)

        # rotor dynamic state
        self.theta = float(self.turbine["initial_angle"])
        self.omega = float(self.turbine["initial_angular_velocity"])

        self.inlet_cfg = cfg["air"]["inlet"]
        self.inlet_dir = np.asarray(self.inlet_cfg["velocity"], dtype=float)
        self.inlet_speed = float(np.linalg.norm(self.inlet_dir))
        self.t_end = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])
        self.dt = min(0.008, self.sample_dt / 2.0)

        self.smoke_cfg = cfg["smoke"]
        self.smoke_rate = float(self.smoke_cfg.get("source_rate_each", 1.0))
        self.smoke_radius = float(self.smoke_cfg.get("source_radius", 0.1))
        self.tracer_gain = 60.0  # tracer mass rate scaling (kg/s per source)
        self.emission_start = float(self.smoke_cfg.get("emission_start", 0.0))
        self.emission_end = float(self.smoke_cfg.get("emission_end", self.t_end))
        self.channels = list(self.smoke_cfg.get("channels", ["smoke_red", "smoke_green", "smoke_ochre"]))
        tips = np.asarray(self.smoke_cfg["rotor_local_tip_positions"], dtype=float)
        self.tip_local = tips.copy()
        dirs = tips.copy()
        dirs[:, 0] = 0.0
        self.blade_dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

        self._load_rotor_asset()
        self._build_rotor_model()
        self._allocate_fields()
        self._setup_boundaries()
        self._reset_budget()

        self.time = 0.0
        self.steps = 0
        self.aero_torque = 0.0
        self.friction_torque = 0.0
        self.div_residual = 0.0
        self.umax = 0.0

    # ------------------------------------------------------------ rigid asset
    def _load_rotor_asset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        asset = root / "assets" / "three_blade_rotor.obj"
        verts: list[list[float]] = []
        tris: list[list[int]] = []
        groups: dict[str, list[int]] = {}
        current = "hub"
        for raw in asset.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                low = line.lower()
                if low.startswith("# blade"):
                    current = low.lstrip("# ").strip().replace(" ", "_")
                continue
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
                groups.setdefault(current, []).append(len(verts) - 1)
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[k], idx[k + 1]])
        V = np.asarray(verts, dtype=float)
        T = np.asarray(tris, dtype=np.int64)
        self.mesh_vertices = V
        a = V[T[:, 1]] - V[T[:, 0]]
        b = V[T[:, 2]] - V[T[:, 0]]
        areas = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
        w = np.zeros(len(V))
        for k, tri in enumerate(T):
            w[tri] += areas[k] / 3.0
        self.mesh_areas = w
        mass = float(self.turbine["mass"])
        m = mass * w / w.sum()
        com = (m[:, None] * V).sum(axis=0) / mass
        R = V - com
        I = np.zeros((3, 3))
        for k in range(len(V)):
            r = R[k]
            I += m[k] * ((r @ r) * np.eye(3) - np.outer(r, r))
        self.mesh_mass = mass
        self.mesh_com = com
        self.I_body = I
        # blade cross sections (span/chord), taken from the frozen mesh groups
        spans, chords = [], []
        for name, idx in groups.items():
            if "blade" not in name:
                continue
            P = V[idx]
            r = np.hypot(P[:, 1], P[:, 2])
            order = np.argsort(r)
            rs = r[order]
            cut = int(np.argmax(np.diff(rs))) + 1
            root_sel = order[:cut]
            tip_sel = order[cut:]
            spans.append((float(r[root_sel].mean()), float(r[tip_sel].mean())))
            chords.append((self._section_chord(P[root_sel]), self._section_chord(P[tip_sel])))
        spans = np.asarray(spans)
        chords = np.asarray(chords)
        self.blade_r_root = float(spans[:, 0].mean())
        self.blade_r_tip = float(spans[:, 1].mean())
        self.blade_chord_root = float(chords[:, 0].mean())
        self.blade_chord_tip = float(chords[:, 1].mean())

    @staticmethod
    def _section_chord(P: np.ndarray) -> float:
        """Tangential chord of a blade ring: arc length spanned by its corners."""
        y = P[:, 1]
        z = P[:, 2]
        radius = np.hypot(y, z)
        if radius.size == 0:
            return 0.0
        ang = np.arctan2(z, y)
        spread = float(ang.max() - ang.min())
        if spread > math.pi:
            spread = 2.0 * math.pi - spread
        return float(2.0 * radius.mean() * math.sin(0.5 * spread))

    # ------------------------------------------------------------- blade model
    def _build_rotor_model(self) -> None:
        self.n_blades = int(self.blade_dirs.shape[0])
        r0 = max(self.blade_r_root + 0.05, 0.25)
        r1 = self.blade_r_tip
        self.n_elem = 8
        self.elem_dr = (r1 - r0) / self.n_elem
        self.elem_r = r0 + (np.arange(self.n_elem) + 0.5) * self.elem_dr
        frac = (self.elem_r - self.blade_r_root) / max(r1 - self.blade_r_root, 1e-9)
        self.elem_chord = self.blade_chord_root + frac * (
            self.blade_chord_tip - self.blade_chord_root
        )
        self.pitch = math.radians(12.0)  # blade setting angle (cambered/twisted blade model)
        self.n_element_tot = self.n_blades * self.n_elem
        self.elem_area_tot = np.tile(self.elem_chord * self.elem_dr, self.n_blades)
        self.I_axis = float(self.I_body[0, 0])
        if not np.isfinite(self.I_axis) or self.I_axis <= 0.0:
            self.I_axis = 0.3 * self.mesh_mass * self.blade_r_tip ** 2
        fr = self.turbine["axle_friction"]
        self.fric_b = float(fr["viscous_coefficient"])
        self.fric_c = float(fr["coulomb_torque"])
        self.fric_ws = float(fr["regularization_speed"])
        # gaussian spreading kernel for the actuator line
        self.kernel_eps = max(1.25 * self.h, 0.15)
        rad_cells = int(math.ceil(3.0 * self.kernel_eps / self.h))
        off = np.arange(-rad_cells, rad_cells + 1) * self.h
        OX, OY, OZ = np.meshgrid(off, off, off, indexing="ij")
        self.kern_off = np.stack([OX.ravel(), OY.ravel(), OZ.ravel()], axis=1)
        d2 = np.sum(self.kern_off ** 2, axis=1)
        wgt = np.exp(-d2 / (2.0 * self.kernel_eps ** 2))
        self.kern_off = self.kern_off[wgt > 1e-4]
        wgt = wgt[wgt > 1e-4]
        self.kern_w = wgt / wgt.sum()
        self.kern_cells = self.kern_off / self.h

    def _allocate_fields(self) -> None:
        n = (self.nx, self.ny, self.nz)
        self.u = np.zeros((self.nx + 1, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny + 1, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz + 1))
        self.smoke = np.zeros((3, *n))
        self.force = np.zeros((3, *n))
        self.solid = np.zeros(n, dtype=bool)
        self.cell_mass = np.full(n, self.rho * self.h ** 3)
        self._build_solids()
        # DST-I eigenvalues of the Dirichlet Laplacian
        kx = np.arange(self.nx)[:, None, None]
        ky = np.arange(self.ny)[None, :, None]
        kz = np.arange(self.nz)[None, None, :]
        self.lam = (4.0 / self.h ** 2) * (
            np.sin(math.pi * (kx + 1.0) / (2.0 * (self.nx + 1))) ** 2
            + np.sin(math.pi * (ky + 1.0) / (2.0 * (self.ny + 1))) ** 2
            + np.sin(math.pi * (kz + 1.0) / (2.0 * (self.nz + 1))) ** 2
        )

    def _build_solids(self) -> None:
        """Frozen static solids: tower, nacelle (cylinders) and the ground wall."""
        root = Path(__file__).resolve().parents[1]
        geom_path = root / "scene_geometry.yaml"
        try:
            import yaml

            geo = yaml.safe_load(geom_path.read_text(encoding="utf-8"))
            self.geometry_objects = {o["id"]: o for o in geo["objects"]}
        except Exception:
            self.geometry_objects = {}
        for name in ("tower", "nacelle"):
            obj = self.geometry_objects.get(name)
            if obj is None:
                continue
            g = obj["geometry"]
            t = np.asarray(obj["transform"]["translation"], dtype=float)
            rad = float(g["radius"])
            hgt = float(g["height"])
            rot = np.asarray(obj["transform"].get("rotation_xyzw", [0, 0, 0, 1]), dtype=float)
            axis = self._quat_axis(rot)
            d = self._point_axis_dist(self.CX, self.CY, self.CZ, t, axis)
            along = (
                (self.CX - t[0]) * axis[0]
                + (self.CY - t[1]) * axis[1]
                + (self.CZ - t[2]) * axis[2]
            )
            inside = (d <= rad) & (np.abs(along) <= 0.5 * hgt)
            self.solid |= inside
        self.solid_indices = np.argwhere(self.solid)
        # surface samples on the frozen walls (for the boundary trajectory fields)
        self._build_boundary_samples()

    @staticmethod
    def _point_axis_dist(X, Y, Z, p, axis):
        rx, ry, rz = X - p[0], Y - p[1], Z - p[2]
        proj = rx * axis[0] + ry * axis[1] + rz * axis[2]
        dx = rx - proj * axis[0]
        dy = ry - proj * axis[1]
        dz = rz - proj * axis[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _quat_axis(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        n = math.sqrt(x * x + y * y + z * z + w * w)
        return np.array([x, y, z]) / max(n, 1e-12)

    def _build_boundary_samples(self) -> None:
        pts, nrm = [], []
        # ground patches (no-slip floor)
        for gx in np.linspace(self.bmin[0] + 0.5, self.bmax[0] - 0.5, 5):
            for gy in np.linspace(self.bmin[1] + 0.5, self.bmax[1] - 0.5, 4):
                pts.append([gx, gy, 0.0])
                nrm.append([0.0, 0.0, 1.0])
        # tower / nacelle surfaces
        for name in ("tower", "nacelle"):
            obj = self.geometry_objects.get(name)
            if obj is None:
                continue
            g = obj["geometry"]
            t = np.asarray(obj["transform"]["translation"], dtype=float)
            rad = float(g["radius"])
            hgt = float(g["height"])
            axis = self._quat_axis(np.asarray(obj["transform"].get("rotation_xyzw", [0, 0, 0, 1]), dtype=float))
            e1 = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = e1 - axis * (e1 @ axis)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis, e1)
            for s in (-1.0, 1.0):
                for k, al in enumerate(np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False)):
                    if k % 2 and s < 0:
                        continue
                    n = math.cos(al) * e1 + math.sin(al) * e2
                    pts.append((t + s * 0.5 * hgt * axis + rad * n).tolist())
                    nrm.append(n.tolist())
        self.wall_points = np.asarray(pts, dtype=float)
        self.wall_normals = np.asarray(nrm, dtype=float)

    # ------------------------------------------------------------- boundaries
    def _setup_boundaries(self) -> None:
        self.inlet_face = np.zeros((self.ny, self.nz))
        self.out_x = self.nx
        Zi, Yi = np.meshgrid(self.zc, self.yc, indexing="ij")
        self.face_x_area = self.h * self.h
        # angular momentum flux sampling coordinates (relative to world origin)
        self.sy = self.yc
        self.sz = self.zc

    def inlet_velocity(self, time: float) -> np.ndarray:
        return self.scene.inlet_velocity(time)

    def _apply_boundaries(self, time: float, dt: float) -> None:
        """Inlet ramp, open-face outflow, ground no-slip, solid masking."""
        vin = self.inlet_velocity(time)
        self.u[0, :, :] = vin[0]
        self.v[0, :, :] = vin[1]
        self.w[0, :, :] = vin[2]
        # outflow (zero gradient) on the open faces
        self.u[self.nx, :, :] = self.u[self.nx - 1, :, :]
        self.v[:, 0, :] = self.v[:, 1, :]
        self.v[:, self.ny, :] = self.v[:, self.ny - 1, :]
        self.u[:, 0, :] = self.u[:, 1, :]
        self.u[:, self.ny - 1, :] = self.u[:, self.ny - 2, :]
        for f in (self.u, self.v, self.w):
            f[:, :, 0] = f[:, :, 1]
        self.w[:, :, self.nz] = self.w[:, :, self.nz - 1]
        # ground: no slip / no penetration
        self.u[:, :, 0] = 0.0
        self.v[:, :, 0] = 0.0
        self.w[:, :, 0] = 0.0

    # ----------------------------------------------------------- cell helpers
    def cell_velocity(self):
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return uc, vc, wc

    def _interp_scalar(self, field, ox, oy, oz, px, py, pz):
        nx, ny, nz = field.shape
        fx = (px - ox) / self.h
        fy = (py - oy) / self.h
        fz = (pz - oz) / self.h
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, max(nx - 2, 0))
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, max(ny - 2, 0))
        k0 = np.clip(np.floor(fz).astype(np.int64), 0, max(nz - 2, 0))
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        tz = np.clip(fz - k0, 0.0, 1.0)
        if nx == 1:
            i0 = np.zeros_like(i0)
            tx = np.zeros_like(tx)
        if ny == 1:
            j0 = np.zeros_like(j0)
            ty = np.zeros_like(ty)
        if nz == 1:
            k0 = np.zeros_like(k0)
            tz = np.zeros_like(tz)
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        k1 = np.minimum(k0 + 1, nz - 1)
        f = field
        c00 = f[i0, j0, k0] * (1 - tx) + f[i1, j0, k0] * tx
        c10 = f[i0, j1, k0] * (1 - tx) + f[i1, j1, k0] * tx
        c01 = f[i0, j0, k1] * (1 - tx) + f[i1, j0, k1] * tx
        c11 = f[i0, j1, k1] * (1 - tx) + f[i1, j1, k1] * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    def sample_velocity(self, px, py, pz):
        ux = self._interp_scalar(self.u, self.bmin[0], self.bmin[1] + 0.5 * self.h,
                                 self.bmin[2] + 0.5 * self.h, px, py, pz)
        vy = self._interp_scalar(self.v, self.bmin[0] + 0.5 * self.h, self.bmin[1],
                                 self.bmin[2] + 0.5 * self.h, px, py, pz)
        wz = self._interp_scalar(self.w, self.bmin[0] + 0.5 * self.h,
                                 self.bmin[1] + 0.5 * self.h, self.bmin[2], px, py, pz)
        return ux, vy, wz

    # -------------------------------------------------------------- projection
    def _project(self, dt: float) -> float:
        div = (
            (self.u[1:, :, :] - self.u[:-1, :, :])
            + (self.v[:, 1:, :] - self.v[:, :-1, :])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / self.h
        rhs = div / dt
        phat = sp_fft.dstn(rhs, type=1, axes=(0, 1, 2))
        phat = -phat / self.lam
        p = sp_fft.idstn(phat, type=1, axes=(0, 1, 2))
        gu = np.empty_like(self.u)
        gu[1:self.nx, :, :] = (p[1:, :, :] - p[:-1, :, :]) / self.h
        gu[0, :, :] = p[0, :, :] / self.h
        gu[self.nx, :, :] = -p[self.nx - 1, :, :] / self.h
        gv = np.empty_like(self.v)
        gv[:, 1:self.ny, :] = (p[:, 1:, :] - p[:, :-1, :]) / self.h
        gv[:, 0, :] = p[:, 0, :] / self.h
        gv[:, self.ny, :] = -p[:, self.ny - 1, :] / self.h
        gw = np.empty_like(self.w)
        gw[:, :, 1:self.nz] = (p[:, :, 1:] - p[:, :, :-1]) / self.h
        gw[:, :, 0] = p[:, :, 0] / self.h
        gw[:, :, self.nz] = -p[:, :, self.nz - 1] / self.h
        self.u -= dt * gu
        self.v -= dt * gv
        self.w -= dt * gw
        div2 = (
            (self.u[1:, :, :] - self.u[:-1, :, :])
            + (self.v[:, 1:, :] - self.v[:, :-1, :])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / self.h
        return float(np.abs(div2)[2:-2, 2:-2, 2:-2].max())

    # ------------------------------------------------------------- advection
    def _flatten_index(self, ox, oy, oz, px, py, pz, shape):
        nx, ny, nz = shape
        fx = (px - ox) / self.h
        fy = (py - oy) / self.h
        fz = (pz - oz) / self.h
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        k0 = np.clip(np.floor(fz).astype(np.int64), 0, nz - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        tz = np.clip(fz - k0, 0.0, 1.0)
        base = i0 * (ny * nz) + j0 * nz + k0
        return base, tx, ty, tz

    def _gather(self, flat, base, tx, ty, tz, ny, nz):
        iy = ny * nz
        iz = nz
        w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
        w100 = tx * (1.0 - ty) * (1.0 - tz)
        w010 = (1.0 - tx) * ty * (1.0 - tz)
        w110 = tx * ty * (1.0 - tz)
        w001 = (1.0 - tx) * (1.0 - ty) * tz
        w101 = tx * (1.0 - ty) * tz
        w011 = (1.0 - tx) * ty * tz
        w111 = tx * ty * tz
        return (
            flat[base] * w000
            + flat[base + iy] * w100
            + flat[base + iz] * w010
            + flat[base + iy + iz] * w110
            + flat[base + 1] * w001
            + flat[base + iy + 1] * w101
            + flat[base + iz + 1] * w011
            + flat[base + iy + iz + 1] * w111
        )

    def _departure(self, dt):
        uc, vc, wc = self.cell_velocity()
        self._uc, self._vc, self._wc = uc, vc, wc
        Xd = self.CX - dt * uc
        Yd = self.CY - dt * vc
        Zd = self.CZ - dt * wc
        return Xd, Yd, Zd

    @staticmethod
    def _pad_axis(arr, axis):
        """Average neighbouring samples along ``axis`` (edge values copied)."""
        n = arr.shape[axis]
        shape = list(arr.shape)
        shape[axis] = n + 1
        out = np.empty(shape, dtype=arr.dtype)
        lo = [slice(None)] * 3
        mid = [slice(None)] * 3
        hi = [slice(None)] * 3
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        lo[axis] = 0
        mid[axis] = slice(1, -1)
        hi[axis] = -1
        a[axis] = slice(None, -1)
        b[axis] = slice(1, None)
        out[tuple(lo)] = arr[tuple(lo)]
        out[tuple(mid)] = 0.5 * (arr[tuple(a)] + arr[tuple(b)])
        out[tuple(hi)] = arr[tuple(hi)]
        return out

    def _advect_velocity(self, dt):
        Xd, Yd, Zd = self._departure(dt)
        ny, nz = self.ny, self.nz
        # u faces (x-faces, cell centres in y/z)
        xu = self._pad_axis(Xd, 0)
        yu = self._pad_axis(Yd, 0)
        zu = self._pad_axis(Zd, 0)
        base, tx, ty, tz = self._flatten_index(
            self.bmin[0], self.bmin[1] + 0.5 * self.h, self.bmin[2] + 0.5 * self.h,
            xu, yu, zu, self.u.shape)
        u_new = self._gather(self.u.ravel(), base, tx, ty, tz, ny, nz).reshape(self.u.shape)
        xv = self._pad_axis(Xd, 1)
        yv = self._pad_axis(Yd, 1)
        zv = self._pad_axis(Zd, 1)
        base, tx, ty, tz = self._flatten_index(
            self.bmin[0] + 0.5 * self.h, self.bmin[1], self.bmin[2] + 0.5 * self.h,
            xv, yv, zv, self.v.shape)
        v_new = self._gather(self.v.ravel(), base, tx, ty, tz, ny, nz).reshape(self.v.shape)
        xw = self._pad_axis(Xd, 2)
        yw = self._pad_axis(Yd, 2)
        zw = self._pad_axis(Zd, 2)
        base, tx, ty, tz = self._flatten_index(
            self.bmin[0] + 0.5 * self.h, self.bmin[1] + 0.5 * self.h, self.bmin[2],
            xw, yw, zw, self.w.shape)
        w_new = self._gather(self.w.ravel(), base, tx, ty, tz, ny, nz).reshape(self.w.shape)
        self.u = np.ascontiguousarray(u_new)
        self.v = np.ascontiguousarray(v_new)
        self.w = np.ascontiguousarray(w_new)

    def _minmod_slopes(self, q, axis):
        dm = q - np.roll(q, 1, axis=axis)
        dp = np.roll(q, -1, axis=axis) - q
        s = np.where(dm * dp > 0.0, np.sign(dm) * np.minimum(np.abs(dm), np.abs(dp)), 0.0)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = 0
        hi[axis] = -1
        s[tuple(lo)] = 0.0
        s[tuple(hi)] = 0.0
        return s

    def _scalar_tendency(self, q):
        """-div(u q) with a second order limited upwind flux (inlet air is clean)."""
        rhs = np.zeros_like(q)
        h = self.h
        for axis, vel in ((0, self._uc), (1, self._vc), (2, self._wc)):
            sl = self._minmod_slopes(q, axis)
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(None, -1)
            b[axis] = slice(1, None)
            qL = q[tuple(a)] + 0.5 * sl[tuple(a)]
            qR = q[tuple(b)] - 0.5 * sl[tuple(b)]
            vf = 0.5 * (vel[tuple(a)] + vel[tuple(b)])
            face = np.where(vf > 0.0, qL, qR)
            flux = vf * face
            r_lo = [slice(None)] * 3
            r_in = [slice(None)] * 3
            r_hi = [slice(None)] * 3
            r_lo[axis] = 0
            r_in[axis] = slice(1, -1)
            r_hi[axis] = -1
            left = [slice(None)] * 3
            left[axis] = 0
            right = [slice(None)] * 3
            right[axis] = -1
            q_left = q[tuple(left)]
            q_right = q[tuple(right)]
            u_left = vel[tuple(left)]
            u_right = vel[tuple(right)]
            f_left = np.minimum(u_left, 0.0) * q_left
            f_right = u_right * q_right
            fl = [slice(None)] * 3
            fl[axis] = slice(0, -1)
            fr = [slice(None)] * 3
            fr[axis] = slice(1, None)
            el0 = [slice(None)] * 3
            el1 = [slice(None)] * 3
            el0[axis] = 0
            el1[axis] = -1
            rhs[tuple(r_lo)] += (f_left - flux[tuple(el0)]) / h
            rhs[tuple(r_in)] += (flux[tuple(fl)] - flux[tuple(fr)]) / h
            rhs[tuple(r_hi)] += (flux[tuple(el1)] - f_right) / h
        return rhs

    def _advect_smoke(self, dt):
        for c in range(self.smoke.shape[0]):
            q = self.smoke[c]
            q = q + dt * self._scalar_tendency(q)
            np.clip(q, 0.0, None, out=q)
            self.smoke[c] = q

    def _emit_smoke(self, dt):
        if self.time < self.emission_start - 1e-12 or self.time > self.emission_end:
            return
        Rm = _rot_x(self.theta)
        sig = max(self.smoke_radius, 0.8 * self.h)
        rad = int(math.ceil(3.0 * sig / self.h))
        mass = self.smoke_rate * self.tracer_gain * dt
        off = self.h * np.arange(-rad, rad + 1)
        OX, OY, OZ = np.meshgrid(off, off, off, indexing="ij")
        wgt = np.exp(-(OX ** 2 + OY ** 2 + OZ ** 2) / (2.0 * sig ** 2))
        wgt /= wgt.sum()
        for b in range(min(self.n_blades, self.smoke.shape[0])):
            p = self.hub + Rm @ self.tip_local[b]
            i0 = int(math.floor((p[0] - self.bmin[0]) / self.h))
            j0 = int(math.floor((p[1] - self.bmin[1]) / self.h))
            k0 = int(math.floor((p[2] - self.bmin[2]) / self.h))
            ii = i0 + np.arange(-rad, rad + 1)
            jj = j0 + np.arange(-rad, rad + 1)
            kk = k0 + np.arange(-rad, rad + 1)
            vi = (ii >= 0) & (ii < self.nx)
            vj = (jj >= 0) & (jj < self.ny)
            vk = (kk >= 0) & (kk < self.nz)
            if not (vi.any() and vj.any() and vk.any()):
                continue
            wsub = wgt[np.ix_(vi, vj, vk)]
            self.smoke[b][np.ix_(ii[vi], jj[vj], kk[vk])] += mass * wsub
            self.mass_source[b] += mass

    # ---------------------------------------------------------- rotor physics
    def _friction_torque(self, omega):
        return self.fric_b * omega + self.fric_c * math.tanh(omega / max(self.fric_ws, 1e-9))

    def _blade_kinematics(self, theta):
        Rm = _rot_x(theta)
        cos_p, sin_p = math.cos(self.pitch), math.sin(self.pitch)
        n_tot = self.n_element_tot
        pos = np.zeros((n_tot, 3))
        tan = np.zeros((n_tot, 3))
        nrm = np.zeros((n_tot, 3))
        cdir = np.zeros((n_tot, 3))
        span = np.zeros(n_tot)
        e = 0
        for b in range(self.n_blades):
            sd = Rm @ self.blade_dirs[b]
            td = np.cross(self.axle, sd)
            nv = -sin_p * td + cos_p * self.axle
            cv = cos_p * td + sin_p * self.axle
            for k in range(self.n_elem):
                r = float(self.elem_r[k])
                pos[e] = self.hub + r * sd
                tan[e] = td
                nrm[e] = nv
                cdir[e] = cv
                span[e] = r
                e += 1
        return pos, tan, nrm, cdir, span

    def _blade_forces(self, theta, omega):
        """Blade element loads from the local relative wind (lift + drag polar).

        ``pitch`` is the blade setting angle built into the rotor model; the
        frozen mesh supplies span, taper, chord and inertia.  The same force
        is applied to the rotor and (with opposite sign) to the fluid.
        """
        pos, tan, nrm, cdir, span = self._blade_kinematics(theta)
        ux, uy, uz = self.sample_velocity(pos[:, 0], pos[:, 1], pos[:, 2])
        uf = np.stack([ux, uy, uz], axis=1)
        vb = (omega * span)[:, None] * tan
        wrel = uf - vb
        wmag = np.linalg.norm(wrel, axis=1)
        wc = np.einsum("ij,ij->i", wrel, cdir)
        wn = np.einsum("ij,ij->i", wrel, nrm)
        alpha = np.arctan2(wn, wc)
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        cl = np.clip(2.0 * math.pi * sa, -1.30, 1.30)
        cd = 0.02 + 0.05 * np.abs(sa)
        lift_dir = (-sa)[:, None] * cdir + ca[:, None] * nrm
        safe = np.maximum(wmag, 1e-9)
        drag_dir = wrel / safe[:, None]
        area = self.elem_area_tot
        coeff = 0.5 * self.rho * area * wmag ** 2
        force = coeff[:, None] * (cl[:, None] * lift_dir - cd[:, None] * drag_dir)
        torque_axis = float(np.sum(span * np.einsum("ij,ij->i", force, tan)))
        return pos, force, torque_axis

    def _spread_forces(self, pos, force):
        self.force.fill(0.0)
        nx, ny, nz = self.shape
        cell_vol = self.h ** 3
        base = np.floor((pos - self.bmin[None, :]) / self.h).astype(np.int64)
        n_el = pos.shape[0]
        flat = np.zeros((n_el, self.kern_off.shape[0]), dtype=np.int64)
        offs = np.rint(self.kern_off / self.h).astype(np.int64)
        for e in range(n_el):
            i = base[e, 0] + offs[:, 0]
            j = base[e, 1] + offs[:, 1]
            k = base[e, 2] + offs[:, 2]
            ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
            flat[e] = np.where(ok, i * (ny * nz) + j * nz + k, 0)
        valid = flat >= 0
        for c in range(3):
            w = -force[:, c][:, None] * self.kern_w[None, :] / cell_vol
            acc = np.bincount(flat.ravel(), weights=np.where(valid, w, 0.0).ravel(),
                              minlength=nx * ny * nz)
            self.force[c] = acc.reshape(nx, ny, nz)

    def _apply_body_force(self, dt):
        fx = self.force[0]
        fy = self.force[1]
        fz = self.force[2]
        ux = np.zeros_like(self.u)
        ux[1:-1] = 0.5 * (fx[:-1] + fx[1:])
        ux[0] = fx[0]
        ux[-1] = fx[-1]
        self.u += dt * ux / self.rho
        vy = np.zeros_like(self.v)
        vy[:, 1:-1] = 0.5 * (fy[:, :-1] + fy[:, 1:])
        vy[:, 0] = fy[:, 0]
        vy[:, -1] = fy[:, -1]
        self.v += dt * vy / self.rho
        wz = np.zeros_like(self.w)
        wz[:, :, 1:-1] = 0.5 * (fz[:, :, :-1] + fz[:, :, 1:])
        wz[:, :, 0] = fz[:, :, 0]
        wz[:, :, -1] = fz[:, :, -1]
        self.w += dt * wz / self.rho

    def _solid_reaction(self, dt):
        if not self.solid_indices.size:
            return np.zeros(3), np.zeros(3)
        i = self.solid_indices[:, 0]
        j = self.solid_indices[:, 1]
        k = self.solid_indices[:, 2]
        vol = self.h ** 3
        ucell = 0.5 * (self.u[i, j, k] + self.u[i + 1, j, k])
        vcell = 0.5 * (self.v[i, j, k] + self.v[i, j + 1, k])
        wcell = 0.5 * (self.w[i, j, k] + self.w[i, j, k + 1])
        f = self.rho * vol * np.array([ucell.sum(), vcell.sum(), wcell.sum()]) / dt
        pts = np.stack([self.xc[i], self.yc[j], self.zc[k]], axis=1)
        lever = pts.sum(axis=0) / len(pts)
        self.u[i, j, k] = 0.0
        self.u[i + 1, j, k] = 0.0
        self.v[i, j, k] = 0.0
        self.v[i, j + 1, k] = 0.0
        self.w[i, j, k] = 0.0
        self.w[i, j, k + 1] = 0.0
        return f, lever

    # ---------------------------------------------------------------- budget
    def _flush_flux(self, mom, pos, uout, dt):
        """Accumulate inflow/outflow angular momentum through an open face."""
        m = self.rho * self.h * self.h * dt * uout
        cr = np.cross(pos, mom)
        mp = np.where(m > 0.0, m, 0.0)
        mn = np.where(m < 0.0, -m, 0.0)
        self.out_flux += (cr * mp[..., None]).sum(axis=(0, 1))
        self.in_flux += (cr * mn[..., None]).sum(axis=(0, 1))

    def _fluxes(self, dt):
        uc, vc, wc = self._uc, self._vc, self._wc
        X, Y, Z = self.CX, self.CY, self.CZ
        # x_min (outward -x)
        pos = np.stack([np.full((self.ny, self.nz), self.xf[0]), np.broadcast_to(self.yc[:, None], (self.ny, self.nz)),
                        np.broadcast_to(self.zc[None, :], (self.ny, self.nz))], axis=-1)
        mom = np.stack([self.u[0], vc[0], wc[0]], axis=-1)
        self._flush_flux(mom, pos, -mom[..., 0], dt)
        # x_max (outward +x)
        pos = np.stack([np.full((self.ny, self.nz), self.xf[self.nx]),
                        np.broadcast_to(self.yc[:, None], (self.ny, self.nz)),
                        np.broadcast_to(self.zc[None, :], (self.ny, self.nz))], axis=-1)
        mom = np.stack([self.u[self.nx], vc[-1], wc[-1]], axis=-1)
        self._flush_flux(mom, pos, mom[..., 0], dt)
        # y_min (outward -y)
        pos = np.stack([np.broadcast_to(self.xc[:, None], (self.nx, self.nz)),
                        np.full((self.nx, self.nz), self.yf[0]),
                        np.broadcast_to(self.zc[None, :], (self.nx, self.nz))], axis=-1)
        mom = np.stack([uc[:, 0, :], self.v[:, 0, :], wc[:, 0, :]], axis=-1)
        self._flush_flux(mom, pos, -mom[..., 1], dt)
        # y_max (outward +y)
        pos = np.stack([np.broadcast_to(self.xc[:, None], (self.nx, self.nz)),
                        np.full((self.nx, self.nz), self.yf[self.ny]),
                        np.broadcast_to(self.zc[None, :], (self.nx, self.nz))], axis=-1)
        mom = np.stack([uc[:, -1, :], self.v[:, self.ny, :], wc[:, -1, :]], axis=-1)
        self._flush_flux(mom, pos, mom[..., 1], dt)
        # z_max (open top, outward +z)
        pos = np.stack([np.broadcast_to(self.xc[:, None], (self.nx, self.ny)),
                        np.broadcast_to(self.yc[None, :], (self.nx, self.ny)),
                        np.full((self.nx, self.ny), self.zf[self.nz])], axis=-1)
        mom = np.stack([uc[:, :, -1], vc[:, :, -1], self.w[:, :, self.nz]], axis=-1)
        self._flush_flux(mom, pos, mom[..., 2], dt)

    # ------------------------------------------------------------- time step
    def _reset_budget(self) -> None:
        self.rotor_impulse = np.zeros(3)
        self.fluid_impulse = np.zeros(3)
        self.global_impulse = np.zeros(3)
        self.in_flux = np.zeros(3)
        self.out_flux = np.zeros(3)
        self.mass_source = np.zeros(3)
        self.rotor_force_sum = np.zeros(3)
        self.solid_force_sum = np.zeros(3)
        self.solid_point = self.hub.copy()

    def _step(self, dt: float) -> None:
        uc, vc, wc = self.cell_velocity()
        self._uc, self._vc, self._wc = uc, vc, wc
        umax = max(float(np.abs(uc).max()), float(np.abs(vc).max()),
                   float(np.abs(wc).max()), 1e-9)
        self.umax = umax
        if umax * dt > 0.9 * self.h:
            dt = 0.9 * self.h / umax

        pos, force, tau_aero = self._blade_forces(self.theta, self.omega)
        tau_fric = self._friction_torque(self.omega)
        alpha = (tau_aero - tau_fric) / self.I_axis
        omega_new = self.omega + dt * alpha
        theta_new = self.theta + 0.5 * dt * (self.omega + omega_new)
        self.aero_torque = tau_aero
        self.friction_torque = tau_fric

        self._force_cache = np.array(force)
        self._spread_forces(pos, force)
        self._advect_velocity(dt)
        self._apply_body_force(dt)
        solid_force, solid_point = self._solid_reaction(dt)
        self.solid_force_sum = solid_force
        self.solid_point = solid_point

        t_next = self.time + dt
        self._apply_boundaries(t_next, dt)
        self.div_residual = self._project(dt)
        self._apply_boundaries(t_next, dt)

        self._uc, self._vc, self._wc = self.cell_velocity()
        self._emit_smoke(dt)
        self._advect_smoke(dt)
        self._accumulate_budget(dt, pos, force, tau_fric, solid_force, solid_point)

        self.theta = theta_new
        self.omega = omega_new
        self.time = t_next
        self.steps += 1

    def _accumulate_budget(self, dt, pos, force, tau_fric, solid_force, solid_point):
        self._uc, self._vc, self._wc = self.cell_velocity()
        tau_aero = np.zeros(3)
        for e in range(pos.shape[0]):
            tau_aero += np.cross(pos[e], force[e])
        fsum = force.sum(axis=0)
        self.rotor_force_sum = fsum
        tau_axle = np.cross(self.hub, -fsum)
        self.rotor_impulse += dt * (tau_aero + tau_axle - tau_fric * self.axle)
        self.fluid_impulse += dt * (-tau_aero - np.cross(solid_point, solid_force))
        self.global_impulse += dt * (tau_axle - np.cross(solid_point, solid_force))
        self._fluxes(dt)

    # ---------------------------------------------------------- output frame
    def _blade_samples(self):
        pos, tan, nrm, cdir, span = self._blade_kinematics(self.theta)
        target = (self.omega * span)[:, None] * tan
        return pos, nrm, target

    def _boundary_frames(self):
        pos, nrm, target = self._blade_samples()
        pts = np.vstack([self.wall_points + 0.5 * self.h * self.wall_normals, pos])
        nn = np.vstack([self.wall_normals, nrm])
        tv = np.vstack([np.zeros_like(self.wall_points), target])
        ux, uy, uz = self.sample_velocity(pts[:, 0], pts[:, 1], pts[:, 2])
        measured = np.stack([ux, uy, uz], axis=1)
        return nn, tv, measured, nrm, target, measured[-pos.shape[0]:]

    def _budget_arrays(self, pos, force):
        G = 3
        K = 2 + self.n_element_tot
        ef = np.zeros((G, K, 3))
        fp = np.zeros((G, K, 3))
        ec = np.zeros((G, K, 3))
        fsum = force.sum(axis=0)
        for g in (0, 1, 2):
            if g == 0:
                ef[g, 0] = -fsum
                fp[g, 0] = self.hub
                ec[g, 1] = -self.friction_torque * self.axle
                for e in range(self.n_element_tot):
                    ef[g, 2 + e] = force[e]
                    fp[g, 2 + e] = pos[e]
            elif g == 1:
                ef[g, 0] = -self.solid_force_sum
                fp[g, 0] = self.solid_point
                for e in range(self.n_element_tot):
                    ef[g, 2 + e] = -force[e]
                    fp[g, 2 + e] = pos[e]
            else:
                ef[g, 0] = -fsum
                fp[g, 0] = self.hub
                ef[g, 1] = -self.solid_force_sum
                fp[g, 1] = self.solid_point
        return ef, fp, ec

    def _frame(self) -> StateFrame:
        uc, vc, wc = self.cell_velocity()
        vel = np.stack([uc, vc, wc], axis=-1).astype(np.float32)
        Rm = _rot_x(self.theta)
        I_world = Rm @ self.I_body @ Rm.T
        quat = _quat_x(self.theta)
        pos, nrm, target = self._blade_samples()
        nn, tv, measured, bn, bt, bv = self._boundary_frames()
        fields = {
            "air_density": np.full(self.shape, self.rho, dtype=np.float32),
            "velocity_field": vel,
            "smoke_red": self.smoke[0].astype(np.float32),
            "smoke_green": self.smoke[1].astype(np.float32),
            "smoke_ochre": self.smoke[2].astype(np.float32),
            "grid_origin": self.bmin.astype(float),
            "rotor_local_tip_positions": self.tip_local.astype(float),
            "boundary_normal": nn.astype(float),
            "boundary_target_velocity": tv.astype(float),
            "boundary_velocity": measured.astype(float),
            "blade_boundary_normal": bn.astype(float),
            "blade_boundary_target_velocity": bt.astype(float),
            "blade_boundary_velocity": bv.astype(float),
        }
        ef, fp, ec = self._budget_arrays(pos, self._last_force())
        G, K, _ = ef.shape
        fields["budget_domain_weights"] = np.array([[1.0], [0.0], [1.0]])
        fields["budget_external_force"] = ef
        fields["budget_force_position"] = fp
        fields["budget_external_couple"] = ec
        fields["budget_cumulative_external_angular_impulse"] = np.vstack(
            [self.rotor_impulse, self.fluid_impulse, self.global_impulse])
        fields["budget_cumulative_inflow_angular_momentum"] = np.vstack(
            [self.in_flux, self.in_flux, self.in_flux])
        fields["budget_cumulative_outflow_angular_momentum"] = np.vstack(
            [self.out_flux, self.out_flux, self.out_flux])
        fields["cumulative_mass_source"] = self.mass_source.copy()
        fields["interaction_force"] = np.vstack([self.rotor_force_sum, -self.rotor_force_sum])
        frame = StateFrame(
            time=float(self.time),
            position=self.hub.reshape(1, 3),
            velocity=np.zeros((1, 3)),
            orientation=quat.reshape(1, 4),
            angular_velocity=np.array([[self.omega * self.axle[0], self.omega * self.axle[1],
                                        self.omega * self.axle[2]]]),
            mass=np.array([self.mesh_mass]),
            body_id=np.array([int(self.turbine["rotor_body_id"])], dtype=np.int64),
            attributes={
                "rotor_angle": np.array([self.theta]),
                "inertia_tensor": I_world.reshape(1, 3, 3),
                "mass_quantity": np.array([self.mesh_mass]),
            },
            fields=fields,
            scalars={
                "reference_velocity": float(self.inlet_speed),
                "reference_angular_momentum": float(
                    self.rho * self.inlet_speed * self.blade_r_tip ** 3),
                "grid_spacing": float(self.h),
                "characteristic_length": float(self.blade_r_tip),
                "rotor_angle": float(self.theta),
                "rotor_angular_velocity": float(self.omega),
                "rotor_radius": float(self.blade_r_tip),
                "axle_viscous_friction": float(self.fric_b),
                "axle_coulomb_friction": float(self.fric_c),
                "air_density": float(self.rho),
                "inlet_speed": float(self.inlet_speed),
                "aero_torque": float(self.aero_torque),
                "axle_friction_torque": float(self.friction_torque),
            },
        )
        frame.validate()
        return frame

    def _last_force(self):
        if getattr(self, "_force_cache", None) is None:
            _, f, _ = self._blade_forces(self.theta, self.omega)
            self._force_cache = np.array(f)
        return self._force_cache

    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        if not math.isfinite(t) or t < 0.0:
            raise ValueError("time must be finite and non-negative")
        guard = 0
        while self.time < t - 1e-12:
            self._step(min(self.dt, t - self.time))
            guard += 1
            if guard > 20000:
                break
        return self._frame()

    def diagnostics(self) -> dict:
        return {
            "steps": int(self.steps),
            "time": float(self.time),
            "rotor_angle": float(self.theta),
            "rotor_angular_velocity": float(self.omega),
            "inertia_axis": float(self.I_axis),
            "aero_torque": float(self.aero_torque),
            "friction_torque": float(self.friction_torque),
            "max_divergence": float(self.div_residual),
            "max_velocity": float(self.umax),
            "blade_r_tip": float(self.blade_r_tip),
            "blade_chord_root": float(self.blade_chord_root),
            "blade_chord_tip": float(self.blade_chord_tip),
        }
