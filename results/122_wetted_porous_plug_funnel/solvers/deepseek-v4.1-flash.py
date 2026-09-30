"""Two-way coupled wetted porous plug / funnel solver.

Implemented from scratch for the frozen ``122_wetted_porous_plug_funnel``
scene (no external physics engine).  The deformable porous plug is a
Lagrangian mass-spring lattice built directly from the frozen
``assets/porous_plug.obj`` triangle mesh; the liquid is a weakly compressible
SPH particle system.  Coupling is two-way through a deformable
triangle-surface contact model: the current plug triangles exert
penalty/dashpot tractions on the liquid and receive the equal and opposite
reaction distributed to their nodes with barycentric weights.  Wetting is a
simulated per-node saturation state that lowers the local Young's modulus from
the dry value to ``wet_softening_factor * dry``, so the plug squeezes through
the throat purely because its constitutive response softens -- the geometry is
never shrunk or widened.

Integration is deterministic explicit symplectic (semi-implicit) Euler with a
fixed substep ``dt = sample_dt / 40``; every requested sample time is reached
exactly from the initial state.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# Frozen physical / geometric constants
# --------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
PROFILE_R = (0.31, 0.31, 0.58, 0.88, 1.10)
PROFILE_Z = (0.86, 1.02, 1.48, 1.88, 2.15)
WALL_THICKNESS = 0.045
FLOOR_Z = 0.06
FLOOR_R = 1.35

# --------------------------------------------------------------------------
# Numerical parameters
# --------------------------------------------------------------------------
SUBSTEPS_PER_SAMPLE = 80
DT = 0.02 / SUBSTEPS_PER_SAMPLE          # 2.5e-4 s
SPH_H = 0.16                             # SPH smoothing length [m]
RHO0 = 1000.0                            # liquid rest density [kg/m^3]
CS = 25.0                                # reduced numerical sound speed
EQ_GAMMA = 7.0
ALPHA_VISC = 0.08
WATER_SPACING = 0.08
R_SKIN_LIQUID = 0.030                    # plug/liquid contact range [m]
K_SOLID_LIQUID = 4.0e3                   # N/m per liquid particle
C_SOLID_LIQUID = 1.0                     # N*s/m normal dashpot per particle
K_WALL = 2.0e4                           # N/m node-wall penalty
ZETA_WALL = 0.35
ZETA_BOND = 0.18
R_WET = 0.12                             # wetting detection radius [m]
K_WET = 2.5                              # 1/s wetting rate
D_WET = 0.25                             # pore imbibition diffusivity [m^2/s]
MAX_SPEED = 15.0                         # numerical safety clamp [m/s]

MASS_SPLIT = 2                           # conserved material columns
DOMAIN_COUNT = 3                         # plug, liquid, global
FORCE_TERMS = 3                          # gravity, funnel wall, catch floor


def _parse_obj(path: Path):
    """Return (vertices[N,3], triangles[T,3]) from the frozen ASCII OBJ."""
    vertices = []
    faces = []
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            loop = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            for k in range(1, len(loop) - 1):
                faces.append([loop[0], loop[k], loop[k + 1]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    """Undirected unique edges of a triangle soup."""
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(raw, axis=1)
    return np.unique(edges, axis=0)


def _poly6(r2: np.ndarray) -> np.ndarray:
    h = SPH_H
    out = np.zeros_like(r2)
    mask = r2 < h * h
    q = (h * h - r2[mask]) ** 3
    out[mask] = 315.0 / (64.0 * np.pi * h ** 9) * q
    return out


def _spiky_grad(r: np.ndarray) -> np.ndarray:
    """dW/dr for the spiky kernel (negative, zero outside the support)."""
    h = SPH_H
    out = np.zeros_like(r)
    mask = r < h
    out[mask] = -45.0 / (np.pi * h ** 6) * (h - r[mask]) ** 2
    return out


def _closest_point_on_triangles(P, A, B, C):
    """Vectorised closest point + barycentric weights on all (P, triangle) pairs.

    P: [M,3]; A,B,C: [T,3].  Returns cp [M,T,3] and bary [M,T,3] (A,B,C).
    """
    P = P[:, None, :]
    A = A[None, :, :]
    B = B[None, :, :]
    C = C[None, :, :]
    m, t = P.shape[0], A.shape[1]
    ab = B - A
    ac = C - A
    ap = P - A
    d1 = np.einsum("mtk,mtk->mt", ab, ap)
    d2 = np.einsum("mtk,mtk->mt", ac, ap)
    bp = P - B
    d3 = np.einsum("mtk,mtk->mt", ab, bp)
    d4 = np.einsum("mtk,mtk->mt", ac, bp)
    cpv = P - C
    d5 = np.einsum("mtk,mtk->mt", ab, cpv)
    d6 = np.einsum("mtk,mtk->mt", ac, cpv)

    cp = np.zeros((m, t, 3))
    bary = np.zeros((m, t, 3))
    A_b = np.broadcast_to(A, (m, t, 3))
    B_b = np.broadcast_to(B, (m, t, 3))
    C_b = np.broadcast_to(C, (m, t, 3))

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    cp[mask] = A_b[mask]
    bary[mask] = (1.0, 0.0, 0.0)

    mask = (d3 >= 0.0) & (d4 <= d3)
    cp[mask] = B_b[mask]
    bary[mask] = (0.0, 1.0, 0.0)

    mask = (d6 >= 0.0) & (d5 <= d6)
    cp[mask] = C_b[mask]
    bary[mask] = (0.0, 0.0, 1.0)

    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    den = np.where(mask, d1 - d3, 1.0)
    v = np.where(mask, d1 / np.where(den == 0.0, 1.0, den), 0.0)
    cp[mask] = (A_b + v[:, :, None] * ab)[mask]
    bary[..., 0][mask] = 1.0 - v[mask]
    bary[..., 1][mask] = v[mask]

    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    den = np.where(mask, d2 - d6, 1.0)
    w = np.where(mask, d2 / np.where(den == 0.0, 1.0, den), 0.0)
    cp[mask] = (A_b + w[:, :, None] * ac)[mask]
    bary[..., 0][mask] = 1.0 - w[mask]
    bary[..., 2][mask] = w[mask]

    va = d3 * d6 - d5 * d4
    mask = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    den = np.where(mask, (d4 - d3) + (d5 - d6), 1.0)
    w = np.where(mask, (d4 - d3) / np.where(den == 0.0, 1.0, den), 0.0)
    cp[mask] = (B_b - w[:, :, None] * ab + w[:, :, None] * ac)[mask]
    bary[..., 1][mask] = 1.0 - w[mask]
    bary[..., 2][mask] = w[mask]

    denom = va + vb + vc
    safe = np.abs(denom) < 1e-300
    denom = np.where(safe, 1.0, denom)
    vv = vb / denom
    ww = vc / denom
    mask = ~(safe | (d1 <= 0.0) & (d2 <= 0.0) | (d3 >= 0.0) & (d4 <= d3)
             | (d6 >= 0.0) & (d5 <= d6) | (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
             | (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
             | (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0))
    cp[mask] = (A_b + vv[:, :, None] * ab + ww[:, :, None] * ac)[mask]
    bary[..., 0][mask] = 1.0 - vv[mask] - ww[mask]
    bary[..., 1][mask] = vv[mask]
    bary[..., 2][mask] = ww[mask]
    return cp, bary


class BenchmarkSolver:
    """Deterministic two-way coupled poro-elasto-capillary funnel solver."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        plug_cfg = cfg["plug"]
        liquid_cfg = cfg["liquid"]
        self.dt = DT

        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "porous_plug.obj"
        verts, tris = _parse_obj(mesh_path)
        edges = _unique_edges(tris)
        self.n_plug = len(verts)
        self.tri = tris

        # liquid particle lattice (conserves the frozen liquid mass)
        lo = np.asarray(liquid_cfg["bounds_min"], dtype=float)
        hi = np.asarray(liquid_cfg["bounds_max"], dtype=float)
        axes = [np.arange(a + 0.5 * WATER_SPACING, b, WATER_SPACING) for a, b in zip(lo, hi)]
        water = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self.n_water = len(water)
        self.n = self.n_plug + self.n_water

        self.pos = np.vstack((verts, water))
        self.vel = np.zeros((self.n, 3))
        self.vel[self.n_plug:] = np.asarray(liquid_cfg["initial_velocity"], dtype=float)

        m_plug = float(plug_cfg["mass"]) / self.n_plug
        m_water = float(liquid_cfg["density"]) * WATER_SPACING ** 3
        self.plug_mass = m_plug
        self.water_mass = m_water
        self.mass = np.concatenate((np.full(self.n_plug, m_plug),
                                    np.full(self.n_water, m_water)))
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.concatenate((np.ones(self.n_plug, dtype=np.int64),
                                       np.full(self.n_water, 2, dtype=np.int64)))
        self.plug_slice = slice(0, self.n_plug)
        self.water_slice = slice(self.n_plug, self.n)

        # elastic lattice
        self.bond_i = edges[:, 0]
        self.bond_j = edges[:, 1]
        self.bond_l0 = np.linalg.norm(self.pos[self.bond_i] - self.pos[self.bond_j], axis=1)
        self.volume0 = np.pi * float(plug_cfg["radius"]) ** 2 * float(plug_cfg["height"])
        self.area_eff = (self.volume0 / self.n_plug) ** (2.0 / 3.0)
        self.e_dry = float(plug_cfg["dry_young_modulus"])
        self.wet_factor = float(plug_cfg["wet_softening_factor"])
        self.saturation = np.zeros(self.n_plug)
        self.young = np.full(self.n_plug, self.e_dry)

        # conserved material columns: [plug material, liquid material]
        self.mass_quantity = np.zeros((self.n, MASS_SPLIT))
        self.mass_quantity[: self.n_plug, 0] = m_plug
        self.mass_quantity[self.n_plug:, 1] = m_water
        self.phase_id = np.concatenate((np.zeros(self.n_plug), np.ones(self.n_water)))
        self.porosity = np.concatenate((np.full(self.n_plug, float(plug_cfg["porosity"])),
                                        np.zeros(self.n_water)))
        self.cumulative_mass_source = np.zeros(MASS_SPLIT)

        # SPH equation-of-state stiffness
        self.eos_b = RHO0 * CS ** 2 / EQ_GAMMA
        self.spiky_grad = _spiky_grad

        # budget bookkeeping
        self.impulse_ext = np.zeros((DOMAIN_COUNT, 3))
        self.inflow_momentum = np.zeros((DOMAIN_COUNT, 3))
        self.outflow_momentum = np.zeros((DOMAIN_COUNT, 3))
        self.domain_weights = np.zeros((DOMAIN_COUNT, self.n))
        self.domain_weights[0, : self.n_plug] = 1.0
        self.domain_weights[1, self.n_plug:] = 1.0
        self.domain_weights[2, :] = 1.0
        self.length_scale = float(np.max(self.pos[:, 2]) - np.min(self.pos[:, 2]))
        total_mass = float(self.mass.sum())
        self.reference_momentum = float(total_mass * np.sqrt(9.81 * self.length_scale))

        # seat the plug on the frozen funnel before t = 0 (the frozen mesh is
        # authored wider than the cone and must come to rest in contact)
        self._settle()

        # initial (t = 0) external / interaction bookkeeping
        self._ext_pp = np.zeros((self.n, 3))
        self._ext_pp[:, 2] = self.mass * GRAVITY[2]
        self._ext_budget = np.zeros((DOMAIN_COUNT, FORCE_TERMS, 3))
        self._ext_budget[0, 0, 2] = m_plug * self.n_plug * GRAVITY[2]
        self._ext_budget[1, 0, 2] = m_water * self.n_water * GRAVITY[2]
        self._ext_budget[2, 0, 2] = (m_plug * self.n_plug + m_water * self.n_water) * GRAVITY[2]
        self._interaction = np.zeros((3, 3))

        self._t = 0.0
        self.frames = 0
        self.diag = {
            "n_particles": int(self.n),
            "n_plug_nodes": int(self.n_plug),
            "n_liquid_particles": int(self.n_water),
            "dt": float(self.dt),
            "substeps_per_sample": int(SUBSTEPS_PER_SAMPLE),
        }

    def _settle(self) -> None:
        """Quasi-static seating of the dry plug on the funnel cone."""
        dt = self.dt
        pos = self.pos[self.plug_slice]
        vel = np.zeros_like(pos)
        m = self.plug_mass
        for _ in range(4000):
            nvec, pen = self._wall_contact(pos)
            f = self._bond_forces(pos, vel, self.young)
            f[:, 2] += m * GRAVITY[2]
            f += self._contact_force(nvec, pen, vel, m)
            vel = 0.85 * vel + f * (dt / m)
            pos += vel * dt
            nvec, pen = self._wall_contact(pos)
            pos += pen[:, None] * nvec
        self.pos[self.plug_slice] = pos
        self.vel[: self.n_plug] = 0.0

    def _bond_forces(self, pos, vel, young):
        ib, jb = self.bond_i, self.bond_j
        d = pos[jb] - pos[ib]
        length = np.linalg.norm(d, axis=1)
        safe = np.maximum(length, 1e-12)
        nrm = d / safe[:, None]
        e = 0.5 * (young[ib] + young[jb])
        k = self.area_eff * e / self.bond_l0
        vrel = vel[jb] - vel[ib]
        vn = np.einsum("ij,ij->i", vrel, nrm)
        c = 2.0 * ZETA_BOND * np.sqrt(np.maximum(k, 0.0) * (0.5 * self.plug_mass))
        fmag = k * (length - self.bond_l0) + c * vn
        f = fmag[:, None] * nrm
        out = np.zeros_like(pos)
        np.add.at(out, ib, f)
        np.add.at(out, jb, -f)
        return out

    def _wall_contact(self, pos):
        """Signed penetration and unit push direction for the funnel shell."""
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = np.hypot(x, y)
        rs = np.maximum(r, 1e-9)
        zi = np.clip(z, PROFILE_Z[0], PROFILE_Z[-1])
        r_in = np.interp(zi, PROFILE_Z, PROFILE_R)
        r_out = r_in + WALL_THICKNESS
        idx = np.clip(np.searchsorted(PROFILE_Z, zi, side="right") - 1, 0, len(PROFILE_Z) - 2)
        slope = ((PROFILE_R[idx + 1] - PROFILE_R[idx])
                 / (PROFILE_Z[idx + 1] - PROFILE_Z[idx]))
        norm = np.sqrt(1.0 + slope * slope)
        n_r = -1.0 / norm
        n_z = slope / norm
        nvec = np.stack((n_r * x / rs, n_r * y / rs, n_z), axis=1)
        d_in = r - r_in
        d_out = r_out - r
        in_slab = (d_in > 0.0) & (d_out > 0.0)
        push_in = d_in <= d_out
        pen = np.where(in_slab & push_in, d_in,
                       np.where(in_slab, d_out, 0.0))
        inside = (z >= PROFILE_Z[0]) & (z <= PROFILE_Z[-1])
        pen = np.where(inside, pen, 0.0)
        sgn = np.where(push_in, 1.0, -1.0)
        nvec = nvec * np.where(inside & in_slab, sgn, 0.0)[:, None]
        return nvec, pen

    def _floor_contact_pen(self, pos):
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = np.hypot(x, y)
        pen = np.where((z < FLOOR_Z) & (r < FLOOR_R), FLOOR_Z - z, 0.0)
        return pen

    def _contact_force(self, nvec, pen, vel, mass):
        vn = np.einsum("ij,ij->i", vel, nvec)
        c = 2.0 * ZETA_WALL * np.sqrt(K_WALL * mass)
        fmag = K_WALL * pen + c * np.maximum(-vn, 0.0) * (pen > 0.0)
        return fmag[:, None] * nvec

    # ------------------------------------------------------------------
    # one physical substep
    # ------------------------------------------------------------------
    def _sph_water_forces(self, pos_w, vel_w):
        """Weakly compressible SPH pressure + artificial viscosity forces."""
        n_w = len(pos_w)
        m = self.water_mass
        tree = cKDTree(pos_w)
        pairs = tree.query_pairs(SPH_H, output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros((n_w, 3))
        i, j = pairs[:, 0], pairs[:, 1]
        dvec = pos_w[j] - pos_w[i]
        r2 = np.einsum("ij,ij->i", dvec, dvec)
        r = np.sqrt(np.maximum(r2, 1e-18))
        nrm = dvec / r[:, None]
        dW = self.spiky_grad(r)
        w0 = _poly6(np.zeros(1))[0]
        rho = np.full(n_w, m * w0)
        np.add.at(rho, i, m * _poly6(r2))
        np.add.at(rho, j, m * _poly6(r2))
        inv_rho = 1.0 / np.maximum(rho, 0.5 * RHO0)
        pressure = np.maximum(self.eos_b * ((rho / RHO0) ** EQ_GAMMA - 1.0), 0.0)
        pterm = pressure * inv_rho * inv_rho
        fac = -(pterm[i] + pterm[j]) * m * dW
        vij = vel_w[j] - vel_w[i]
        vr = np.einsum("ij,ij->i", vij, nrm)
        mu = SPH_H * vr / (r2 + 0.01 * SPH_H * SPH_H)
        rho_bar = 0.5 * (rho[i] + rho[j])
        pi_v = np.where(vr < 0.0, -ALPHA_VISC * CS * mu / np.maximum(rho_bar, 0.5 * RHO0), 0.0)
        fac = fac + (-m * pi_v * dW)
        contrib = fac[:, None] * nrm
        acc = np.zeros((n_w, 3))
        np.add.at(acc, i, contrib)
        np.add.at(acc, j, -contrib)
        return acc * m

    def _plug_liquid_contact(self, pos_w, vel_w):
        """Deformable plug triangle surface vs. liquid.

        Returns (liquid_forces[Nw,3], node_forces[Nplug,3], summed_wall_force).
        Fluid pressure pushes the plug (reaction distributed barycentrically)
        and the moving plug surface pushes the fluid back.
        """
        n_w = len(pos_w)
        liquid_force = np.zeros((n_w, 3))
        node_force = np.zeros((self.n_plug, 3))
        plug_pos = self.pos[self.plug_slice]
        vel_p = self.vel[self.plug_slice]
        centre = plug_pos.mean(axis=0)
        radius = float(np.max(np.linalg.norm(plug_pos - centre, axis=1))) + 0.5 * WATER_SPACING
        near = np.linalg.norm(pos_w - centre, axis=1) < radius
        if not np.any(near):
            return liquid_force, node_force
        psel = np.flatnonzero(near)
        pts = pos_w[psel]
        a = plug_pos[self.tri[:, 0]]
        b = plug_pos[self.tri[:, 1]]
        c = plug_pos[self.tri[:, 2]]
        cp, bary = _closest_point_on_triangles(pts, a, b, c)
        ab = b - a
        ac = c - a
        nvec = np.cross(ab, ac)
        nlen = np.linalg.norm(nvec, axis=1)
        safe = np.maximum(nlen, 1e-14)
        nhat = nvec / safe[:, None]
        signed = np.einsum("mtk,mtk->mt", pts[:, None, :] - cp, nhat[None, :, :])
        mask = (signed < R_SKIN_LIQUID) & (nlen[None, :] > 1e-12)
        fpart = np.zeros_like(pts)
        fmag = np.where(mask, K_SOLID_LIQUID * (R_SKIN_LIQUID - signed), 0.0)
        # relative normal velocity damping (only when approaching the surface)
        vel_nodes = (bary[..., 0][..., None] * vel_p[self.tri[:, 0]][None, :, :]
                     + bary[..., 1][..., None] * vel_p[self.tri[:, 1]][None, :, :]
                     + bary[..., 2][..., None] * vel_p[self.tri[:, 2]][None, :, :])
        vrel = vel_w[psel][:, None, :] - vel_nodes
        vn = np.einsum("mtk,mtk->mt", vrel, nhat[None, :, :])
        damp = C_SOLID_LIQUID * np.maximum(-vn, 0.0) * mask
        total = (fmag + damp)[..., None] * nhat[None, :, :]
        total = np.where(mask[..., None], total, 0.0)
        fpart = total.sum(axis=1)
        liquid_force[psel] = fpart
        vals = -total * bary
        idx = np.broadcast_to(self.tri[None, :, :], bary.shape)
        np.add.at(node_force, idx.reshape(-1), vals.reshape(-1, 3))
        return liquid_force, node_force

    def _step(self) -> None:
        dt = self.dt
        pos, vel, mass, inv = self.pos, self.vel, self.mass, self.inv_mass
        np_ = self.n_plug
        force = np.zeros_like(pos)
        force[:, 2] += mass * GRAVITY[2]
        ext = np.zeros((DOMAIN_COUNT, FORCE_TERMS, 3))
        g_plug = mass[:np_].sum() * GRAVITY[2]
        g_water = mass[np_:].sum() * GRAVITY[2]
        ext[0, 0, 2] = g_plug
        ext[1, 0, 2] = g_water
        ext[2, 0, 2] = g_plug + g_water

        # elastic lattice (interior to the plug)
        f_el = self._bond_forces(pos[:np_], vel[:np_], self.young)
        force[:np_] += f_el
        elastic_net = f_el.sum(axis=0)

        # rigid funnel shell + catch floor boundary contacts
        nvec, pen = self._wall_contact(pos)
        cw = 2.0 * ZETA_WALL * np.sqrt(K_WALL * mass)
        vn = np.einsum("ij,ij->i", vel, nvec)
        f_wall = (K_WALL * pen + cw * np.maximum(-vn, 0.0) * (pen > 0.0))[:, None] * nvec
        force += f_wall
        ext[0, 1] += f_wall[:np_].sum(axis=0)
        ext[1, 1] += f_wall[np_:].sum(axis=0)
        ext[2, 1] += f_wall.sum(axis=0)

        fpen = self._floor_contact_pen(pos)
        if np.any(fpen > 0.0):
            nfloor = np.zeros_like(pos)
            nfloor[:, 2] = 1.0
            vnf = vel[:, 2]
            f_floor = (K_WALL * fpen + cw * np.maximum(-vnf, 0.0) * (fpen > 0.0))[:, None] * nfloor
            force += f_floor
            ext[0, 2] += f_floor[:np_].sum(axis=0)
            ext[1, 2] += f_floor[np_:].sum(axis=0)
            ext[2, 2] += f_floor.sum(axis=0)

        # external (non-coupled) per-particle force, for the standard field
        ext_pp = force.copy()

        # liquid SPH interior forces
        f_sph = self._sph_water_forces(pos[np_:], vel[np_:])
        force[np_:] += f_sph

        # two-way deformable plug / liquid coupling
        f_liq, f_node = self._plug_liquid_contact(pos[np_:], vel[np_:])
        force[np_:] += f_liq
        force[:np_] += f_node
        water_on_plug = f_node.sum(axis=0)
        plug_on_water = f_liq.sum(axis=0)

        # symplectic Euler integration with a numerical safety clamp
        vel += force * inv[:, None] * dt
        speed = np.linalg.norm(vel, axis=1)
        hot = speed > MAX_SPEED
        if np.any(hot):
            vel[hot] *= (MAX_SPEED / speed[hot])[:, None]
        pos += vel * dt

        # cumulative external impulse per domain (all external terms, once)
        for g in range(DOMAIN_COUNT):
            self.impulse_ext[g] += ext[g].sum(axis=0) * dt

        # wetting front: liquid contact source + capillary imbibition through
        # the porous lattice -> constitutive softening
        d2 = ((pos[np_:][None, :, :] - pos[:np_][:, None, :]) ** 2).sum(axis=-1)
        cnt = (d2 < R_WET * R_WET).sum(axis=0)
        contact = np.minimum(1.0, cnt / 6.0)
        sat = self.saturation
        lap = np.zeros(self.n_plug)
        pair_d = pos[self.bond_i] - pos[self.bond_j]
        pair_l2 = np.maximum(np.einsum("ij,ij->i", pair_d, pair_d), 1e-12)
        flux = (sat[self.bond_j] - sat[self.bond_i]) / pair_l2
        np.add.at(lap, self.bond_i, flux)
        np.add.at(lap, self.bond_j, -flux)
        sat = sat + dt * (K_WET * contact * (1.0 - sat) + D_WET * lap)
        self.saturation = np.clip(sat, 0.0, 1.0)
        self.young = self.e_dry * (1.0 - self.saturation * (1.0 - self.wet_factor))

        self._ext_pp = ext_pp
        self._ext_budget = ext
        self._interaction = np.vstack((elastic_net, water_on_plug, plug_on_water))

    # ------------------------------------------------------------------
    # trajectory output
    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        attrs = {
            "phase_id": self.phase_id,
            "saturation": np.concatenate((self.saturation, np.ones(self.n_water))),
            "porosity": self.porosity,
            "mass_quantity": self.mass_quantity,
        }
        fields = {
            "external_force": self._ext_pp,
            "interaction_force": self._interaction,
            "budget_external_force": self._ext_budget,
            "budget_cumulative_external_impulse": self.impulse_ext,
            "budget_cumulative_inflow_momentum": self.inflow_momentum,
            "budget_cumulative_outflow_momentum": self.outflow_momentum,
            "budget_domain_weights": self.domain_weights,
            "cumulative_mass_source": self.cumulative_mass_source,
        }
        scalars = {
            "reference_momentum": self.reference_momentum,
            "reference_velocity": float(np.sqrt(9.81 * self.length_scale)),
            "time_step": float(self.dt),
            "throat_radius": 0.31,
            "wet_fraction": float(self.saturation.max()),
            "mean_young_modulus": float(self.young.mean()),
        }
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id,
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        steps = int(round((target - self._t) / self.dt))
        if steps < 0:
            steps = 0
        for _ in range(steps):
            self._step()
        self._t += steps * self.dt
        self.frames += 1
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        out = dict(self.diag)
        pz = self.pos[self.plug_slice, 2]
        out.update({
            "final_time": float(self._t),
            "frames_emitted": int(self.frames),
            "plug_centroid_z": float(pz.mean()),
            "plug_min_z": float(pz.min()),
            "max_saturation": float(self.saturation.max()),
            "final_young_modulus": float(self.young.mean()),
            "passed_below_throat": bool(pz.max() < 1.02),
        })
        return out
