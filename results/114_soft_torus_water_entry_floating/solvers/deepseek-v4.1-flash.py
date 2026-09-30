"""Two-way coupled soft-torus water-entry solver.

Fluid
    Weakly compressible SPH on the scene's water particle samples (cubic
    spline kernel, Tait equation of state, Monaghan artificial viscosity).
    The tub walls are impermeable and static.

Solid
    The triangular torus mesh is a mass-spring deformable body: membrane
    springs along the mesh edges, cross-tube springs that keep the tube
    cross-section open, viscous edge damping and a damped volume projection.

Coupling
    The fluid pressure field is interpolated onto the current torus surface
    and integrated as a traction (this reproduces Archimedes buoyancy for a
    closed surface).  A linear drag traction exchanges momentum with the
    fluid.  A short range penalty keeps fluid particles out of the solid.
    Every reaction is applied back onto the nearby fluid particles with the
    SPH kernel, so the pair of interaction forces is exactly equal and
    opposite and the exported momentum budgets match the discrete dynamics.

Everything is deterministic: fixed step count, no randomness, no I/O.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
GRAVITY = np.array([0.0, 0.0, -9.81])

RHO0 = 1000.0                 # water rest density
H = 0.05                      # SPH smoothing length
C0 = 12.0                     # numerical speed of sound
B_EOS = RHO0 * C0 * C0 / 7.0  # Tait stiffness
ALPHA = 0.08                  # artificial viscosity coefficient
SIGMA = 1.0 / (math.pi * H ** 3)
W0 = SIGMA                    # W(0)

SUB_DT = 6.25e-4              # internal step (32 substeps per 0.02 s sample)
SKIN = 0.02                   # neighbour-list skin
REBUILD_EVERY = 5             # steps between neighbour-list rebuilds

# tub interior (inner faces of the static walls)
XMIN, XMAX = -0.50, 0.50
YMIN, YMAX = -0.50, 0.50
ZMIN = 0.0

# soft torus
MAJOR_R = 0.255
MINOR_R = 0.105
N_MAJOR = 48
N_MINOR = 20
TUBE_VOLUME = 2.0 * math.pi ** 2 * MAJOR_R * MINOR_R ** 2

K_EDGE = 300.0                # membrane spring stiffness  [N/m]
K_TUBE = 200.0                # cross-tube spring stiffness [N/m]
C_EDGE = 3.0                  # edge damping [N s/m]
P_SOFT = 15000.0              # smooth cap on the surface traction pressure [Pa]
K_VOLUME = 0.25               # volume projection relaxation
MAX_VOLUME_STEP = 1.0e-4      # per-step scale correction clamp

C_DRAG = 1.0                  # linear drag coefficient [1/s]
K_REP = 3000.0                # fluid-solid penalty stiffness [N/m]
C_REP = 12.0                  # fluid-solid penalty damping [N s/m]
D_REP = 0.02                  # penalty range [m]

DX = 0.04
DEN_FULL = 1.0 / DX ** 3      # full kernel support density (scene particle spacing)

REFERENCE_VELOCITY = math.sqrt(2.0 * 9.81 * 0.40)


def _kernel(q: np.ndarray):
    """Cubic spline kernel shape f(q) and f'(q), support q<2."""
    inner = q < 1.0
    outer = q < 2.0
    w = np.where(inner, 1.0 - 1.5 * q * q + 0.75 * q * q * q,
                 np.where(outer, 0.25 * (2.0 - q) ** 3, 0.0))
    dw = np.where(inner, -3.0 * q + 2.25 * q * q,
                  np.where(outer, -0.75 * (2.0 - q) ** 2, 0.0))
    return w, dw


def _torus_topology():
    """Vertices, triangles and spring edges of the rest torus."""
    verts = []
    for i in range(N_MAJOR):
        u = 2.0 * math.pi * i / N_MAJOR
        for j in range(N_MINOR):
            v = 2.0 * math.pi * j / N_MINOR
            r = MAJOR_R + MINOR_R * math.cos(v)
            verts.append([r * math.cos(u), r * math.sin(u), MINOR_R * math.sin(v)])
    verts = np.asarray(verts, dtype=float)
    faces = []
    for i in range(N_MAJOR):
        for j in range(N_MINOR):
            a = i * N_MINOR + j
            b = ((i + 1) % N_MAJOR) * N_MINOR + j
            c = ((i + 1) % N_MAJOR) * N_MINOR + (j + 1) % N_MINOR
            d = i * N_MINOR + (j + 1) % N_MINOR
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces = np.asarray(faces, dtype=np.int64)

    edges = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for p, q in ((a, b), (b, c), (c, a)):
            edges.add((p, q) if p < q else (q, p))
    cross = []
    for i in range(N_MAJOR):
        for j in range(N_MINOR // 2):
            a = i * N_MINOR + j
            b = i * N_MINOR + (j + N_MINOR // 2) % N_MINOR
            cross.append((a, b))
    return verts, faces, np.asarray(sorted(edges), dtype=np.int64), np.asarray(cross, dtype=np.int64)


def _wall_samples():
    """Boundary samples (position, outward unit normal) on the tub inner faces."""
    pts = []
    nrm = []
    lin = np.linspace(-0.45, 0.45, 10)
    zs = np.linspace(0.05, 0.35, 4)
    for y in lin:
        for x in lin:
            pts.append([x, y, 0.0])
            nrm.append([0.0, 0.0, 1.0])
    for z in zs:
        for y in lin:
            pts.append([XMIN, y, z]); nrm.append([1.0, 0.0, 0.0])
            pts.append([XMAX, y, z]); nrm.append([-1.0, 0.0, 0.0])
    for z in zs:
        for x in lin:
            pts.append([x, YMAX, z]); nrm.append([0.0, -1.0, 0.0])
            pts.append([x, YMIN, z]); nrm.append([0.0, 1.0, 0.0])
    return np.asarray(pts, dtype=float), np.asarray(nrm, dtype=float)


class BenchmarkSolver:
    """Deterministic WCSPH + mass-spring soft torus with two-way coupling."""

    # ------------------------------------------------------------ lifecycle
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self.position = np.array(frame.position, dtype=float, copy=True)
        self.velocity = np.array(frame.velocity, dtype=float, copy=True)
        self.mass = np.array(frame.mass, dtype=float, copy=True)
        self.body_id = np.array(frame.body_id, dtype=np.int32, copy=True)
        self.rest_position = self.position.copy()
        self.n_particles = len(self.position)

        self.t_idx = np.flatnonzero(self.body_id == 1)
        self.f_idx = np.flatnonzero(self.body_id == 2)
        self.n_t = len(self.t_idx)
        self.n_f = len(self.f_idx)

        verts, faces, edges, cross = _torus_topology()
        self.verts0 = verts
        self.faces = faces
        self.edges = edges
        self.cross = cross
        self.edge_l0 = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        self.cross_l0 = np.linalg.norm(verts[cross[:, 0]] - verts[cross[:, 1]], axis=1)
        self.volume0 = self._mesh_volume(self.rest_position[self.t_idx])

        self.f_mass = self.mass[self.f_idx]
        self.t_mass = self.mass[self.t_idx]
        self.m_water = float(np.sum(self.f_mass))
        self.m_torus = float(np.sum(self.t_mass))
        self.m_total = self.m_water + self.m_torus

        self.wall_pos, self.wall_nrm = _wall_samples()
        self.b = len(self.wall_pos)
        self.zeros_b3 = np.zeros((self.b, 3))
        self.zeros_g3 = np.zeros((3, 3))

        # budget bookkeeping: domains 0=water, 1=torus, 2=combined
        self.g = 3
        self.k_terms = 3  # gravity, wall traction, fluid-solid interaction
        self.domain_weights = np.zeros((self.g, self.n_particles))
        self.domain_weights[0, self.f_idx] = 1.0
        self.domain_weights[1, self.t_idx] = 1.0
        self.domain_weights[2, :] = 1.0
        self.cum_external_impulse = np.zeros((self.g, 3))
        self.cum_inflow = np.zeros((self.g, 3))
        self.cum_outflow = np.zeros((self.g, 3))

        self.time = 0.0
        self.step_index = 0
        self.sample_index = 0
        self.sample_times = [float(t) for t in scene.sample_times()]
        self.dt_sample = float(scene.sample_times()[1] - scene.sample_times()[0])
        self.last_force = np.zeros((self.g, self.k_terms, 3))
        self.last_interaction = np.zeros((2, 3))

        self._pairs = None
        self._pairs_age = 0
        self._reset_interval()

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _mesh_volume(pt: np.ndarray) -> float:
        a = pt[:, 0]
        b = pt[:, 1]
        c = pt[:, 2]
        return float(np.sum(a * np.cross(b, c)) / 6.0)

    def _surface_geometry(self, pt: np.ndarray):
        tri = pt[self.faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        nvec = np.cross(e1, e2)
        norm = np.linalg.norm(nvec, axis=1)
        area = 0.5 * norm
        unit = nvec / np.maximum(norm, 1e-12)[:, None]
        vtx_n = np.zeros_like(pt)
        vtx_a = np.zeros(len(pt))
        for k in range(3):
            np.add.at(vtx_n, self.faces[:, k], unit * area[:, None])
            np.add.at(vtx_a, self.faces[:, k], area / 3.0)
        vtx_n /= np.maximum(np.linalg.norm(vtx_n, axis=1), 1e-12)[:, None]
        return vtx_n, vtx_a

    # -------------------------------------------------------------- springs
    def _spring_forces(self, pos, vel, pairs, l0, k, c):
        idx = self.t_idx
        d = pos[idx[pairs[:, 0]]] - pos[idx[pairs[:, 1]]]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        dirv = d / length[:, None]
        vn = np.sum((vel[idx[pairs[:, 0]]] - vel[idx[pairs[:, 1]]]) * dirv, axis=1)
        mag = -k * (length - l0) - c * vn
        f = mag[:, None] * dirv
        out = np.zeros((self.n_t, 3))
        np.add.at(out, pairs[:, 0], f)
        np.add.at(out, pairs[:, 1], -f)
        return out

    # ------------------------------------------------------------ neighbours
    def _rebuild_pairs(self, pf: np.ndarray) -> None:
        tree = cKDTree(pf)
        pairs = tree.query_pairs(2.0 * H + SKIN, output_type="ndarray")
        if len(pairs) == 0:
            self._pairs = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), tree)
            return
        ii = np.concatenate([pairs[:, 0], pairs[:, 1]])
        jj = np.concatenate([pairs[:, 1], pairs[:, 0]])
        self._pairs = (ii.astype(np.int64), jj.astype(np.int64), tree)
        self._pairs_age = 0

    # -------------------------------------------------------------- stepping
    def _reset_interval(self) -> None:
        self.imp_grav = np.zeros((self.g, 3))
        self.imp_wall = np.zeros((self.g, 3))
        self.imp_int = np.zeros((self.g, 3))

    def _wall_project(self, pos, vel, mask):
        """Clamp to the impermeable static tub; return the absorbed impulse."""
        imp = np.zeros(3)
        p = pos[mask]
        v = vel[mask]
        m = self.mass[mask]
        for axis, lo, hi in ((0, XMIN, XMAX), (1, YMIN, YMAX), (2, ZMIN, None)):
            if lo is not None:
                hit = p[:, axis] < lo
                if np.any(hit):
                    out = -v[hit, axis]
                    corr = np.where(out > 0.0, out, 0.0)
                    imp[axis] += float(np.sum(m[hit] * corr))
                    v[hit, axis] += corr
                    p[hit, axis] = lo
            if hi is not None:
                hit = p[:, axis] > hi
                if np.any(hit):
                    out = v[hit, axis]
                    corr = np.where(out > 0.0, out, 0.0)
                    imp[axis] -= float(np.sum(m[hit] * corr))
                    v[hit, axis] -= corr
                    p[hit, axis] = hi
        pos[mask] = p
        vel[mask] = v
        return imp

    def _sph_acceleration(self, pos, vel, rho, pres):
        ii, jj, _ = self._pairs
        m = self.mass[self.f_idx]
        pf = pos[self.f_idx]
        vf = vel[self.f_idx]
        acc = np.zeros((self.n_f, 3))
        if len(ii):
            dr = pf[ii] - pf[jj]
            r2 = np.sum(dr * dr, axis=1) + 1e-18
            r = np.sqrt(r2)
            q = r / H
            w, dw = _kernel(q)
            dwdr = SIGMA * dw / H
            rho_i = rho[ii]
            rho_j = rho[jj]
            term = pres[ii] / (rho_i * rho_i) + pres[jj] / (rho_j * rho_j)
            vij = vf[ii] - vf[jj]
            rv = np.sum(vij * dr, axis=1)
            mu = H * rv / (r2 + 0.01 * H * H)
            mu = np.where(rv < 0.0, mu, 0.0)
            rho_bar = 0.5 * (rho_i + rho_j)
            pie = -ALPHA * C0 * mu / rho_bar
            scale = -(m[ii] * m[jj] * (term + pie) * dwdr / r)
            fx = scale * dr[:, 0]
            fy = scale * dr[:, 1]
            fz = scale * dr[:, 2]
            acc[:, 0] = np.bincount(ii, weights=fx, minlength=self.n_f)
            acc[:, 1] = np.bincount(ii, weights=fy, minlength=self.n_f)
            acc[:, 2] = np.bincount(ii, weights=fz, minlength=self.n_f)
            acc /= m[:, None]
        return acc

    def _coupling(self, pos, vel, rho, pres, tree):
        """Fluid-solid tractions plus the kernel-distributed reactions."""
        pt = pos[self.t_idx]
        vt = vel[self.t_idx]
        pf = pos[self.f_idx]
        vf = vel[self.f_idx]
        f_t = np.zeros((self.n_t, 3))
        f_f = np.zeros((self.n_f, 3))
        vtx_n, vtx_a = self._surface_geometry(pt)

        # --- pressure + drag tractions integrated on the torus surface -----
        lists = tree.query_ball_point(pt, 2.0 * H)
        counts = np.fromiter((len(lst) for lst in lists), dtype=np.int64, count=self.n_t)
        if int(counts.sum()) > 0:
            flat = np.concatenate([np.asarray(lst, dtype=np.int64) for lst in lists])
            vid = np.repeat(np.arange(self.n_t, dtype=np.int64), counts)
            dr = pt[vid] - pf[flat]
            r = np.linalg.norm(dr, axis=1)
            w, _ = _kernel(r / H)
            den = np.bincount(vid, weights=w, minlength=self.n_t)
            den_safe = np.maximum(den, 1e-12)
            p_v = np.bincount(vid, weights=w * pres[flat], minlength=self.n_t) / den_safe
            v_v = np.empty((self.n_t, 3))
            for k in range(3):
                v_v[:, k] = np.bincount(vid, weights=w * vf[flat, k],
                                        minlength=self.n_t) / den_safe
            cov = np.clip(den / DEN_FULL, 0.0, 1.0)
            p_use = P_SOFT * np.tanh(np.maximum(p_v, 0.0) / P_SOFT)
            f_press = -vtx_a[:, None] * p_use[:, None] * vtx_n
            f_drag = -vtx_a[:, None] * RHO0 * C_DRAG * cov[:, None] * (vt - v_v)
            f_t += f_press + f_drag
            share = (w / den_safe[vid])[:, None]
            react = -(f_press[vid] + f_drag[vid]) * share
            for k in range(3):
                f_f[:, k] += np.bincount(flat, weights=react[:, k], minlength=self.n_f)

        # --- short range penalty keeps fluid out of the solid --------------
        tree_t = cKDTree(pt)
        _, nearest = tree_t.query(pf)
        n_v = vtx_n[nearest]
        s = np.sum((pf - pt[nearest]) * n_v, axis=1)
        # the nearest surface point's outward normal points away from the solid
        # material, so the penalty always pushes along +n
        pen = D_REP - s
        active = pen > 0.0
        if np.any(active):
            vrel = np.sum((vf - vt[nearest]) * n_v, axis=1)
            pen_eff = np.minimum(pen, 2.0 * D_REP)
            mag = np.maximum(K_REP * pen_eff - C_REP * vrel, 0.0)
            mag = np.where(active, mag, 0.0)
            f_rep = mag[:, None] * n_v
            f_f += f_rep
            np.add.at(f_t, nearest, -f_rep)
        return f_f, f_t

    def _volume_project(self, pos) -> None:
        pt = pos[self.t_idx]
        vol = self._mesh_volume(pt)
        err = (self.volume0 - vol) / self.volume0
        scale = 1.0 + float(np.clip(K_VOLUME * err, -MAX_VOLUME_STEP, MAX_VOLUME_STEP))
        if abs(scale - 1.0) < 1e-15:
            return
        m = self.t_mass
        center = np.sum(pt * m[:, None], axis=0) / self.m_torus
        pos[self.t_idx] = center + (pt - center) * scale

    def _substep(self, dt: float) -> None:
        pos = self.position
        vel = self.velocity
        pf = pos[self.f_idx]

        if self._pairs is None or self._pairs_age >= REBUILD_EVERY:
            self._rebuild_pairs(pf)
        else:
            self._pairs_age += 1
        ii, jj, tree = self._pairs

        # ---- SPH density and pressure ------------------------------------
        m_f = self.f_mass
        if len(ii):
            dr = pf[ii] - pf[jj]
            r = np.sqrt(np.sum(dr * dr, axis=1) + 1e-18)
            w, _ = _kernel(r / H)
            rho = np.bincount(ii, weights=m_f[jj] * w, minlength=self.n_f) + m_f * W0
        else:
            rho = m_f * W0
        rho = np.maximum(rho, 0.1 * RHO0)
        pres = np.maximum(B_EOS * ((rho / RHO0) ** 7 - 1.0), 0.0)

        # ---- fluid forces -------------------------------------------------
        acc_f = self._sph_acceleration(pos, vel, rho, pres) + GRAVITY

        # ---- solid forces -------------------------------------------------
        f_t = self._spring_forces(pos, vel, self.edges, self.edge_l0, K_EDGE, C_EDGE)
        f_t += self._spring_forces(pos, vel, self.cross, self.cross_l0, K_TUBE, C_EDGE)
        f_t += self.t_mass[:, None] * GRAVITY[None, :]

        f_f_couple, f_t_couple = self._coupling(pos, vel, rho, pres, tree)
        acc_f += f_f_couple / m_f[:, None]
        f_t += f_t_couple

        # ---- impulse bookkeeping (exact for the discrete dynamics) --------
        total_f_couple = np.sum(f_f_couple, axis=0)
        total_t_couple = np.sum(f_t_couple, axis=0)
        self.imp_grav[0] += self.m_water * GRAVITY * dt
        self.imp_grav[1] += self.m_torus * GRAVITY * dt
        self.imp_grav[2] += self.m_total * GRAVITY * dt
        self.imp_int[0] += total_f_couple * dt
        self.imp_int[1] += total_t_couple * dt

        # ---- integrate ----------------------------------------------------
        vel[self.f_idx] += acc_f * dt
        vel[self.t_idx] += f_t / self.t_mass[:, None] * dt
        pos += vel * dt

        imp_w = self._wall_project(pos, vel, self.f_idx)
        imp_t = self._wall_project(pos, vel, self.t_idx)
        self.imp_wall[0] += imp_w
        self.imp_wall[1] += imp_t
        self.imp_wall[2] += imp_w + imp_t

        self._volume_project(pos)

    # ---------------------------------------------------------------- frames
    def _make_frame(self, time: float) -> StateFrame:
        dt_s = self.dt_sample
        force = np.zeros((self.g, self.k_terms, 3))
        force[:, 0, :] = self.imp_grav / dt_s
        force[:, 1, :] = self.imp_wall / dt_s
        force[:, 2, :] = self.imp_int / dt_s
        self.cum_external_impulse += self.imp_grav + self.imp_wall + self.imp_int
        interaction = np.zeros((2, 3))
        interaction[0] = self.imp_int[0] / dt_s
        interaction[1] = self.imp_int[1] / dt_s
        self.last_force = force
        self.last_interaction = interaction

        mass_quantity = self.mass.copy()
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass_quantity,
                "rest_position": self.rest_position.copy(),
            },
            fields={
                "boundary_normal": self.wall_nrm.copy(),
                "boundary_velocity": self.zeros_b3.copy(),
                "boundary_target_velocity": self.zeros_b3.copy(),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": force.copy(),
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
                "interaction_force": interaction.copy(),
                "cumulative_mass_source": np.zeros(2),
            },
            scalars={
                "reference_velocity": float(REFERENCE_VELOCITY),
                "reference_momentum": float(self.m_total * REFERENCE_VELOCITY),
            },
        )
        self._reset_interval()
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / SUB_DT))
        while self.step_index < target:
            self._substep(SUB_DT)
            self.step_index += 1
            self.time = self.step_index * SUB_DT
        self.sample_index += 1
        return self._make_frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "wcsp_soft_torus_two_way",
            "fluid_particles": int(self.n_f),
            "solid_vertices": int(self.n_t),
            "internal_dt": float(SUB_DT),
            "substeps": int(self.step_index),
            "final_time": float(self.time),
            "water_mass": float(self.m_water),
            "torus_mass": float(self.m_torus),
        }
