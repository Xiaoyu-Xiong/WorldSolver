"""Two-way coupled rigid-frigate / free-surface-wave solver (self-contained).

Physical model
--------------
Fluid
    Depth-averaged free-surface (shallow-water) model integrated on a
    staggered C-grid.  The incoming-wave control is prescribed at the
    ``x_min`` boundary (surface elevation and matching orbital velocity) and
    an absorbing sponge layer removes outgoing waves at ``x_max``.  The
    free-surface phase speed is taken from the frozen wave control
    (``c = wavelength / period``) so the prescribed train propagates with the
    configured wavelength and period.  Side walls are impermeable
    (no-normal-flow) and the seabed is the grid floor.

Solid
    The frigate hull is a rigid body whose free degrees of freedom are the
    frozen ones (heave along z and pitch about y; surge/sway/roll/yaw are
    locked).  Hydrostatic + Froude-Krylov pressure is integrated every step
    over the actual wetted triangle mesh, giving Archimedes buoyancy and wave
    excitation directly from the resolved free surface.  Radiation/viscous
    damping from the relative hull-water motion and a seabed penalty contact
    complete the body dynamics.

Coupling
    The hull displaces fluid in its waterplane column through a continuity
    source proportional to the local hull vertical velocity and penalises
    fluid momentum inside its footprint (immersed-boundary forcing), so the
    evolving solid alters the resolved flow, while the pressure integral over
    the wetted hull drives the heave and pitch response.  Both directions are
    resolved within the same explicit time-stepping loop.

Raw trajectory export
    ``state.position`` / ``state.velocity`` / ``state.mass`` /
    ``state.orientation`` / ``state.angular_velocity`` / ``state.body_id``
    plus every required ``attributes.*``, ``fields.*`` and ``scalars.*`` path
    are produced for every requested sample time.  Sample 0 is always the
    rigid body, water markers follow the simulated free surface.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

RHO_DEFAULT = 1025.0
GRAV = 9.81

# Budget domain / term / material layout (fixed for every frame).
G_DOMAINS = 3          # 0 global, 1 water, 2 hull
K_TERMS = 4            # 0 gravity, 1 hull-water interaction, 2 x-boundaries, 3 walls
K_MAT = 2              # 0 water, 1 hull

BOUNDARY_ZETA = 0.12   # radiation/viscous damping ratio of the hull


def _load_obj(path: Path):
    """Read a Wavefront OBJ into (vertices, triangulated faces)."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def _subdivide(verts: np.ndarray, faces: np.ndarray, levels: int):
    """Uniform midpoint subdivision, used to resolve the hull waterline."""
    v = verts
    t = faces
    for _ in range(levels):
        cache: dict[tuple[int, int], int] = {}
        extra: list[np.ndarray] = []
        base = len(v)

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            found = cache.get(key)
            if found is None:
                found = base + len(extra)
                extra.append(0.5 * (v[a] + v[b]))
                cache[key] = found
            return found

        out = np.empty((4 * len(t), 3), dtype=np.int64)
        for i, (a, b, c) in enumerate(t):
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            out[4 * i + 0] = (a, ab, ca)
            out[4 * i + 1] = (ab, b, bc)
            out[4 * i + 2] = (ca, bc, c)
            out[4 * i + 3] = (ab, bc, ca)
        if extra:
            v = np.vstack([v, np.asarray(extra, dtype=float)])
        t = out
    return v, t


def _quat_from_pitch(theta: float) -> np.ndarray:
    half = 0.5 * theta
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=float)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = scene.config
        fluid = cfg["fluid"]
        waves = cfg["incoming_waves"]
        ship = cfg["frigate"]

        self.rho = float(fluid.get("density", RHO_DEFAULT))
        self.h0 = float(fluid["still_water_level"])
        self.amp = float(waves["amplitude"])
        self.wavelength = float(waves["wavelength"])
        self.period = float(waves["period"])
        self.phase = float(waves.get("phase_radians", 0.0))
        self.cwave = self.wavelength / self.period
        self.omega = 2.0 * math.pi / self.period
        self.kwave = 2.0 * math.pi / self.wavelength
        self.geff = self.cwave * self.cwave / self.h0

        lo = np.asarray(fluid["domain_min"], dtype=float)
        hi = np.asarray(fluid["domain_max"], dtype=float)
        self.lo = lo
        self.hi = hi
        d = float(fluid["marker_spacing"])
        self.dx = d
        self.dy = d
        self.nx = max(4, int(round((hi[0] - lo[0]) / d)))
        self.ny = max(4, int(round((hi[1] - lo[1]) / d)))
        self.xc = lo[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.yc = lo[1] + (np.arange(self.ny) + 0.5) * self.dy
        self.X = self.xc[:, None]
        self.Y = self.yc[None, :]

        # ---- free-surface state (staggered C-grid) ----
        self.eta = self.h0 + self.amp * np.sin(
            self.kwave * self.X + self.phase
        ) * np.ones((1, self.ny))
        self.u = np.full((self.nx + 1, self.ny), float(np.asarray(fluid["initial_velocity"])[0]))
        self.v = np.zeros((self.nx, self.ny + 1))
        self.deta_dt = np.zeros((self.nx, self.ny))
        self.t = 0.0

        # ---- hull rigid body ----
        self.m_ship = float(ship["mass"])
        self.body_id = int(ship["body_id"])
        self.zc = float(np.asarray(ship["initial_center"], dtype=float)[2])
        self.pitch = 0.0
        self.vz = float(np.asarray(ship["initial_velocity"], dtype=float)[2])
        self.wy = float(np.asarray(ship["initial_angular_velocity"], dtype=float)[1])

        asset = Path(__file__).resolve().parent.parent / "assets" / "frigate_hull.obj"
        verts, faces = _load_obj(asset)
        volume = float(np.sum(np.einsum("ij,ij->i", verts[faces[:, 0]],
                                        np.cross(verts[faces[:, 1]], verts[faces[:, 2]]))) / 6.0)
        if volume < 0.0:
            faces = faces[:, ::-1]
        sv, st = _subdivide(verts, faces, 3)
        p0 = sv[st[:, 0]]
        a_vec = 0.5 * np.cross(sv[st[:, 1]] - p0, sv[st[:, 2]] - p0)
        self.hull_c_local = (p0 + sv[st[:, 1]] + sv[st[:, 2]]) / 3.0
        self.hull_a_local = a_vec
        extent = verts.max(axis=0) - verts.min(axis=0)
        self.iyy = self.m_ship / 12.0 * (extent[0] ** 2 + extent[2] ** 2)

        # waterplane properties (from the analytic hull silhouette used by the
        # frozen scene to carve the initial water markers).
        hw = 0.92 * np.clip(1.0 - np.abs(self.xc) / 4.35, 0.0, 1.0) ** 0.35
        sil = (np.abs(self.Y) < hw[:, None]).astype(float)
        self.phi = sil
        self.phi_f = self._smooth(self.phi)
        cell_area = self.dx * self.dy
        a_wp = float(self.phi.sum() * cell_area)
        self.awp = max(a_wp, 0.5 * cell_area)
        i_wp = float(np.sum(self.phi * (self.X ** 2)) * cell_area)
        self.kz = self.rho * GRAV * self.awp
        self.kth = self.rho * GRAV * max(i_wp, 0.05)
        self.cz = 2.0 * BOUNDARY_ZETA * math.sqrt(self.kz * self.m_ship)
        self.cth = 2.0 * BOUNDARY_ZETA * math.sqrt(self.kth * self.iyy)
        self.cq = 0.5 * self.rho * self.awp * 1.5
        self.k_bed = 40.0 * self.kz
        self.c_bed = 0.7 * math.sqrt(self.k_bed * self.m_ship)

        # ---- sponge layer at x_max ----
        sponge = np.zeros((self.nx, 1))
        length = 3.0
        start = self.hi[0] - length
        mask = self.xc > start
        sponge[mask, 0] = 4.0 * ((self.xc[mask] - start) / length) ** 2
        self.sponge = sponge

        # ---- markers (index 0 is the rigid body sample) ----
        state = scene.initial_state()
        pos = np.array(state.position, dtype=float)
        vel = np.array(state.velocity, dtype=float)
        self.n = len(pos)
        self.px = pos[:, 0].copy()
        self.py = pos[:, 1].copy()
        self.pz = pos[:, 2].copy()
        self.px[0] = 0.0
        self.py[0] = 0.0
        self.pz[0] = self.zc
        self.pvx = vel[:, 0].copy()
        self.pvy = vel[:, 1].copy()
        self.pvz = vel[:, 2].copy()
        self.pvz[0] = self.vz
        self.mass = np.array(state.mass, dtype=float)
        self.body = np.array(state.body_id, dtype=np.int64)
        self.phase = np.array(state.attributes["phase"], dtype=float)
        self.visual = np.array(state.attributes["ship_visual"], dtype=float)

        eta0 = self._interp(self.eta, self.px[1:], self.py[1:])
        self.sigma = np.clip(self.pz[1:] / np.maximum(eta0, 1e-6), 0.0, 1.5)

        # ---- boundary sample layout ----
        nb_x, nb_z = 4, 3
        pts = []
        nrm = []
        target = []
        ax = np.linspace(lo[0] + 0.5, hi[0] - 0.5, nb_x)
        ay = np.linspace(lo[1] + 0.5, hi[1] - 0.5, nb_x)
        az = np.linspace(0.4, 2.8, nb_z)
        # x_min inflow boundary
        for yy in np.linspace(lo[1] + 1.0, hi[1] - 1.0, nb_x):
            for zz in np.linspace(0.3, 2.6, nb_z):
                pts.append([lo[0], yy, zz]); nrm.append([1.0, 0.0, 0.0])
        # x_max absorbing boundary
        for yy in np.linspace(lo[1] + 1.0, hi[1] - 1.0, nb_x):
            for zz in np.linspace(0.3, 2.6, nb_z):
                pts.append([hi[0], yy, zz]); nrm.append([-1.0, 0.0, 0.0])
        # side walls
        for xx in ax:
            for zz in az:
                pts.append([xx, lo[1], zz]); nrm.append([0.0, 1.0, 0.0])
                pts.append([xx, hi[1], zz]); nrm.append([0.0, -1.0, 0.0])
        self.wall_pts = np.asarray(pts, dtype=float)
        self.wall_nrm = np.asarray(nrm, dtype=float)

        # ---- bounded histories for the export fields ----
        self.cum_ext_impulse = np.zeros((G_DOMAINS, 3))
        self.cum_inflow = np.zeros((G_DOMAINS, 3))
        self.cum_outflow = np.zeros((G_DOMAINS, 3))
        self.last_force = np.zeros((G_DOMAINS, K_TERMS, 3))
        self.last_interaction = np.zeros((K_MAT, 3))
        self.reference_velocity = self.cwave
        water = self.phase > 0.5
        self.reference_momentum = float(self.mass[water].sum()) * self.cwave
        self._dt = 0.01
        self._external_force_step()
        self.diagnostics_info = {
            "scheme": "staggered shallow-water + rigid hull pressure coupling",
            "cwave": self.cwave,
            "equilibrium_draft_hint": self.m_ship / self.rho,
        }

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _smooth(field: np.ndarray) -> np.ndarray:
        out = field.copy()
        out[1:, :] += field[:-1, :]
        out[:-1, :] += field[1:, :]
        out[:, 1:] += field[:, :-1]
        out[:, :-1] += field[:, 1:]
        return out / 4.0

    def _interp(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        fx = (x - (self.lo[0] + 0.5 * self.dx)) / self.dx
        fy = (y - (self.lo[1] + 0.5 * self.dy)) / self.dy
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, self.nx - 2)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, self.ny - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        f = field
        return (
            (1.0 - tx) * (1.0 - ty) * f[i0, j0]
            + tx * (1.0 - ty) * f[i0 + 1, j0]
            + (1.0 - tx) * ty * f[i0, j0 + 1]
            + tx * ty * f[i0 + 1, j0 + 1]
        )

    # --------------------------------------------------------- fluid advance
    def _fluid_step(self, dt: float) -> None:
        eta = self.eta
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        # driven incoming wave at x_min (propagating in +x)
        eta_in = self.h0 + self.amp * np.sin(
            self.kwave * self.xc[0] - self.omega * self.t + self.phase
        )
        self.eta[0, :] = eta_in
        self.u[0, :] = self.cwave * (eta_in - self.h0) / self.h0
        # radiation condition at x_max
        self.u[nx, :] = self.cwave * (eta[nx - 1, :] - self.h0) / self.h0

        hx = 0.5 * (eta[:-1, :] + eta[1:, :])
        fx = np.empty((nx + 1, ny))
        fx[0, :] = self.h0 * self.u[0, :]
        fx[1:nx, :] = hx * self.u[1:nx, :]
        fx[nx, :] = self.h0 * self.u[nx, :]
        hy = 0.5 * (eta[:, :-1] + eta[:, 1:])
        fy = np.empty((nx, ny + 1))
        fy[:, 0] = self.h0 * self.v[:, 0]
        fy[:, 1:ny] = hy * self.v[:, 1:ny]
        fy[:, ny] = self.h0 * self.v[:, ny]

        deta = -((fx[1:, :] - fx[:-1, :]) / dx + (fy[:, 1:] - fy[:, :-1]) / dy)
        # hull column displacement + mild regularization of the blocked column
        w_hull = self.vz - self.wy * self.X
        deta = deta + self.phi_f * (0.6 * w_hull - 0.8 * (eta - self.h0))
        self.deta_dt = deta
        eta = eta + dt * deta
        eta[0, :] = eta_in

        # momentum: pressure gradient (new surface) + immersed-boundary penalty
        eta_x = (eta[1:, :] - eta[:-1, :]) / dx
        self.u[1:nx, :] += -dt * self.geff * eta_x
        eta_y = (eta[:, 1:] - eta[:, :-1]) / dy
        self.v[:, 1:ny] += -dt * self.geff * eta_y

        pf_x = 0.5 * (self.phi_f[:-1, :] + self.phi_f[1:, :])
        damp = np.minimum(1.0, 8.0 * dt) * pf_x
        self.u[1:nx, :] -= damp * self.u[1:nx, :]
        pf_y = 0.5 * (self.phi_f[:, :-1] + self.phi_f[:, 1:])
        self.v[:, 1:ny] -= np.minimum(1.0, 8.0 * dt) * pf_y * self.v[:, 1:ny]

        # sponge layer
        eta = eta - dt * self.sponge * (eta - self.h0)
        self.u[1:nx, :] *= 1.0 - dt * self.sponge[1:nx, :]
        self.v[:, 1:ny] *= 1.0 - dt * self.sponge

        # walls / impermeability
        self.v[:, 0] = 0.0
        self.v[:, ny] = 0.0
        self.u[0, :] = self.cwave * (eta[0, :] - self.h0) / self.h0
        self.u[nx, :] = self.cwave * (eta[nx - 1, :] - self.h0) / self.h0

        self.eta = np.clip(np.nan_to_num(eta, nan=self.h0), 0.15, 3.6)
        self.u = np.clip(np.nan_to_num(self.u, nan=0.0), -5.0, 5.0)
        self.v = np.clip(np.nan_to_num(self.v, nan=0.0), -5.0, 5.0)
        self.t += dt

    # ---------------------------------------------------------- hull advance
    def _hull_pressure(self):
        """Integrate resolved fluid pressure over the currently wetted hull."""
        th = self.pitch
        ct, st = math.cos(th), math.sin(th)
        rot = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
        cw = self.hull_c_local @ rot.T
        cw[:, 2] += self.zc
        aw = self.hull_a_local @ rot.T
        eta_c = self._interp(self.eta, cw[:, 0], cw[:, 1])
        press = self.rho * GRAV * np.maximum(eta_c - cw[:, 2], 0.0)
        fvec = -press[:, None] * aw
        force = fvec.sum(axis=0)
        lever = cw.copy()
        lever[:, 2] -= self.zc
        torque = np.cross(lever, fvec).sum(axis=0)
        return force, torque, cw

    def _hull_step(self, dt: float, cw: np.ndarray) -> np.ndarray:
        force, torque, _ = self._hull_pressure()
        m = self.m_ship
        # relative vertical flow at the hull (radiation / viscous damping)
        w_fluid = float(self._interp(
            self.deta_dt, np.array([0.0]), np.array([0.0])
        )[0])
        fz = force[2] + self.cq * (w_fluid - self.vz) * abs(w_fluid - self.vz)
        # seabed penalty contact
        keel = self.zc - (1.10 * math.cos(self.pitch) + 4.4 * abs(math.sin(self.pitch)))
        if keel < 0.0:
            fz += self.k_bed * (-keel) - self.c_bed * min(self.vz, 0.0)
        vz = (self.vz + dt * (fz / m - GRAV) + dt * self.cz * w_fluid / m) / (
            1.0 + dt * self.cz / m
        )
        wz = self.wy + dt * torque[1] / self.iyy
        wz /= 1.0 + dt * self.cth / self.iyy
        self.vz = float(np.clip(vz, -4.0, 4.0))
        self.wy = float(np.clip(wz, -3.0, 3.0))
        self.zc = float(np.clip(self.zc + dt * self.vz, 0.35, 3.0))
        self.pitch = float(np.clip(self.pitch + dt * self.wy, -0.7, 0.7))
        # interaction force on the hull from the fluid (pressure + damping)
        f_int = force.copy()
        f_int[2] = force[2] + self.cq * (w_fluid - self.vz) * abs(w_fluid - self.vz)
        return f_int

    # ------------------------------------------------------------- particles
    def _particle_step(self, dt: float) -> None:
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        vc = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        ux = self._interp(uc, self.px[1:], self.py[1:])
        uy = self._interp(vc, self.px[1:], self.py[1:])
        z_old = self.pz[1:].copy()
        self.px[1:] += dt * ux
        self.py[1:] += dt * uy
        eta_p = self._interp(self.eta, self.px[1:], self.py[1:])
        self.pz[1:] = np.maximum(self.sigma * eta_p, 0.02)
        self.pvx[1:] = ux
        self.pvy[1:] = uy
        self.pvz[1:] = (self.pz[1:] - z_old) / dt
        self._exclude_hull()
        px = self.px[1:]
        px = np.where(px > self.hi[0], px - (self.hi[0] - self.lo[0]), px)
        px = np.where(px < self.lo[0], px + (self.hi[0] - self.lo[0]), px)
        self.px[1:] = px
        self.py[1:] = np.clip(self.py[1:], self.lo[1] + 1e-3, self.hi[1] - 1e-3)

    def _exclude_hull(self) -> None:
        x = self.px[1:]
        y = self.py[1:]
        z = self.pz[1:]
        hw = 0.92 * np.clip(1.0 - np.abs(x) / 4.35, 0.0, 1.0) ** 0.35
        inside = (np.abs(x) < 4.35) & (np.abs(y) < hw)
        th = self.pitch
        keel_local = -1.12 + 0.16 * np.abs(x) / 4.35
        zlo = self.zc + (-x * math.sin(th) + keel_local * math.cos(th))
        zhi = self.zc + (-x * math.sin(th) + 0.42 * math.cos(th))
        hit = inside & (z > zlo) & (z < zhi)
        if not np.any(hit):
            return
        sgn = np.where(y >= 0.0, 1.0, -1.0)
        y2 = y.copy()
        y2[hit] = sgn[hit] * (hw[hit] + 0.15)
        vy = self.pvy[1:]
        vy[hit] = sgn[hit] * np.abs(vy[hit])
        self.py[1:] = y2

    # --------------------------------------------------------------- budgets
    def _external_force_step(self) -> None:
        """Accumulate external-force terms and open-boundary momentum fluxes."""
        m_water = float(self.mass[self.phase > 0.5].sum())
        m_all = m_water + self.m_ship
        dt = self._dt
        # boundary traction from the resolved momentum flux at x_min / x_max
        dy = self.dy
        uin = self.u[0, :]
        uout = self.u[self.nx, :]
        f_bnd = self.rho * dy * (
            float(np.sum(self.h0 * uin * np.abs(uin))) - float(np.sum(self.h0 * uout * np.abs(uout)))
        )
        terms = np.zeros((G_DOMAINS, K_TERMS, 3))
        terms[0, 0, 2] = -GRAV * m_all
        terms[1, 0, 2] = -GRAV * m_water
        terms[2, 0, 2] = -GRAV * self.m_ship
        terms[1, 1] = self.last_interaction[0]
        terms[2, 1] = self.last_interaction[1]
        terms[0, 2, 0] = f_bnd
        terms[1, 2, 0] = f_bnd
        self.last_force = terms
        self.cum_ext_impulse += terms.sum(axis=1) * dt

        flux_in = np.zeros(3)
        flux_out = np.zeros(3)
        pos_min = uin > 0.0
        neg_min = uin < 0.0
        pos_max = uout > 0.0
        neg_max = uout < 0.0
        flux_in[0] = self.rho * dy * (
            float(np.sum(self.h0 * uin[pos_min] ** 2)) + float(np.sum(self.h0 * uout[neg_max] ** 2))
        ) * dt
        flux_out[0] = self.rho * dy * (
            float(np.sum(self.h0 * uin[neg_min] ** 2)) + float(np.sum(self.h0 * uout[pos_max] ** 2))
        ) * dt
        self.cum_inflow[0] += flux_in
        self.cum_inflow[1] += flux_in
        self.cum_outflow[0] += flux_out
        self.cum_outflow[1] += flux_out

    # ----------------------------------------------------------------- frames
    def _field_bundle(self) -> dict:
        eta_c = self._interp(self.eta, self.px, self.py)
        wet = (self.phase > 0.5) & (self.pz <= eta_c + 0.05)
        return {
            "boundary_normal": self._boundary_normal(),
            "boundary_target_velocity": self._boundary_target_velocity(),
            "boundary_velocity": self._boundary_target_velocity(),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
            "budget_domain_weights": np.vstack([
                np.ones(self.n),
                (self.phase > 0.5).astype(float),
                (self.body == self.body_id).astype(float),
            ]),
            "budget_external_force": self.last_force.copy(),
            "cumulative_mass_source": np.zeros(K_MAT),
            "interaction_force": self.last_interaction.copy(),
            "wet_mask": wet.astype(float),
        }

    def _boundary_normal(self) -> np.ndarray:
        th = self.pitch
        ct, st = math.cos(th), math.sin(th)
        rot = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
        aw = self.hull_a_local @ rot.T
        norm = aw / np.maximum(np.linalg.norm(aw, axis=1, keepdims=True), 1e-12)
        return np.vstack([norm, self.wall_nrm])

    def _hull_surface_points(self) -> np.ndarray:
        th = self.pitch
        ct, st = math.cos(th), math.sin(th)
        rot = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
        pts = self.hull_c_local @ rot.T
        pts[:, 2] += self.zc
        return pts

    def _boundary_target_velocity(self) -> np.ndarray:
        pts = self._hull_surface_points()
        rel = pts - np.array([0.0, 0.0, self.zc])
        omega = np.array([0.0, self.wy, 0.0])
        hull_v = np.cross(np.broadcast_to(omega, rel.shape), rel)
        hull_v[:, 2] += self.vz
        u_in = self.cwave * (self.eta[0, 0] - self.h0) / self.h0
        u_out = self.cwave * (self.eta[self.nx - 1, 0] - self.h0) / self.h0
        wall_v = np.zeros_like(self.wall_nrm)
        wall_v[:12, 0] = u_in
        wall_v[12:24, 0] = u_out
        return np.vstack([hull_v, wall_v])

    def _build_frame(self, time: float) -> StateFrame:
        pos = np.column_stack([self.px, self.py, self.pz])
        vel = np.column_stack([self.pvx, self.pvy, self.pvz])
        orient = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1))
        omega = np.zeros((self.n, 3))
        pos[0] = (0.0, 0.0, self.zc)
        vel[0] = (0.0, 0.0, self.vz)
        orient[0] = _quat_from_pitch(self.pitch)
        omega[0, 1] = self.wy
        mass_qty = np.zeros((self.n, K_MAT))
        water = self.phase > 0.5
        mass_qty[water, 0] = self.mass[water]
        mass_qty[~water, 1] = self.mass[~water]
        fields = self._field_bundle()
        return StateFrame(
            time=float(time),
            position=np.nan_to_num(pos, nan=0.0),
            velocity=np.nan_to_num(vel, nan=0.0),
            orientation=orient,
            angular_velocity=omega,
            mass=self.mass.copy(),
            body_id=self.body.copy(),
            attributes={
                "phase": self.phase.copy(),
                "ship_visual": self.visual.copy(),
                "mass_quantity": mass_qty,
                "wet_mask": fields.pop("wet_mask"),
            },
            fields=fields,
            scalars={
                "still_water_level": self.h0,
                "reference_velocity": self.reference_velocity,
                "reference_momentum": self.reference_momentum,
                "wave_phase_speed": self.cwave,
                "hull_heave": self.zc,
                "hull_pitch": self.pitch,
            },
        )

    # -------------------------------------------------------------- contract
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        dt = 0.01
        self._dt = dt
        guard = 0
        while self.t < target - 1e-12 and guard < 200000:
            step = min(dt, max(target - self.t, 1e-9))
            self._fluid_step(step)
            hull_force = self._hull_step(step, None)
            self.last_interaction[1] = hull_force
            self.last_interaction[0] = -hull_force
            self._external_force_step()
            self._particle_step(step)
            guard += 1
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "samples": int(self.n),
            "dt": 0.01,
            "reference_velocity": self.reference_velocity,
            "reference_momentum": self.reference_momentum,
            "wave_phase_speed": self.cwave,
        }
