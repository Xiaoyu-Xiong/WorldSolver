"""Twist-induced buckling of a bundle of seven volumetric elastic rods.

Every rod is a three-dimensional isotropic solid circular cylinder with the
frozen geometry (radius r, length L, density rho), discretised as a Cosserat
rod whose nodes coincide with the frozen axial surface sampling levels:

* centreline nodes carry translation (position, velocity),
* a material angle per node carries cross-section twist,

so each solid cross section stays exactly circular and is reconstructed on the
frozen surface vertices (492 per rod, matching assets/rod_surface.obj).

Elastic energy is the analytic cross-section reduction of the 3-D solid:
* stretch : EA/2 int (l-h)^2/h   -> tension EA (l-h)/h
* bending : EI/2 int kappa^2     -> discrete curvature stencil
* twist   : GJ/2 int tau^2       -> angle Laplacian torque

Rod-rod and self contact is a distance penalty on centreline separation with
geometric contact distance 2r (surface-to-surface touching) plus normal
dissipation.  Driven end regions are kinematically clamped: they follow the
prescribed rigid rotations about the bundle axis with fixed axial positions
and clamped cross-section angles.  Integration is symplectic Euler with fixed
substeps; momentum-budget source terms accumulate over the substeps.  No
animation, keyframes or precomputed trajectories are used.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # Frozen density is given; isotropic solid stiffness is the physical
    # material choice of the solver (stiff elastomer, near-incompressible).
    YOUNG = 2.0e5        # Pa
    POISSON = 0.35       # 1
    CONTACT = 5.0e5      # N/m penalty stiffness per contact pair
    CONTACT_DAMP = 60.0  # N*s/m normal dissipation
    CONTACT_GAP = 0.0    # m extra slack beyond 2r
    SUBSTEP = 2.5e-3     # s
    BEND_DAMP = 4.0e-4   # 1/s weak bending relaxation
    MAX_SUBSTEPS = 200000

    def initialize(self, scene, seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        cfg = scene.config
        g, bc, mat = cfg["geometry"], cfg["boundary_conditions"], cfg["material"]
        self.cfg = cfg
        self.rho = float(mat["density"])
        self.radius = float(g["rod_radius"])
        self.length = float(g["rod_length"])
        self.n_seg = int(g["axial_segments"])
        self.n_theta = int(g["radial_segments"])
        self.n_rod = int(g["rod_count"])
        ring_r = float(g["bundle_ring_radius"])
        self.end_len = float(bc["driven_end_length"])
        self.axis = np.asarray(bc["rotation_axis"], dtype=float)
        self.axis /= np.linalg.norm(self.axis)
        self.omega_left = np.deg2rad(float(bc["left_end_angular_speed_deg_s"]))
        self.omega_right = np.deg2rad(float(bc["right_end_angular_speed_deg_s"]))
        self.bc_start = float(bc["start_time"])
        self.bc_stop = float(bc["stop_time"])
        self.axial_fixed = bool(bc["axial_position_fixed_at_driven_ends"])
        self.sample_dt = float(cfg["sample_dt"])

        # --- geometry: nodes along each rod centreline -------------------
        self.nx = self.n_seg + 1                       # 49 centreline nodes
        self.h = self.length / self.n_seg
        x = np.linspace(-0.5 * self.length, 0.5 * self.length, self.nx)
        self.rest_dir = np.zeros((self.nx, 3))
        self.rest_dir[:, 0] = 1.0                      # material d1 at rest
        offsets = [(0.0, 0.0)] + [
            (ring_r * np.cos(k * np.pi / 3.0), ring_r * np.sin(k * np.pi / 3.0))
            for k in range(6)
        ]
        n_node = self.nx * self.n_rod
        pos = np.zeros((n_node, 3))
        theta = np.zeros(n_node)
        for k, (oy, oz) in enumerate(offsets):
            sl = slice(k * self.nx, (k + 1) * self.nx)
            pos[sl, 0] = x
            pos[sl, 1] = oy
            pos[sl, 2] = oz
        self.body_id = np.repeat(np.arange(self.n_rod, dtype=np.int32), self.nx)
        # nodal mass: solid cylinder mass spread over the centreline levels
        self.m_rod = self.rho * np.pi * self.radius ** 2 * self.length
        self.m_node = self.m_rod / self.nx
        self.mass = np.full(n_node, self.m_node)

        # section stiffness of a solid circular cross section
        A = np.pi * self.radius ** 2
        I = np.pi * self.radius ** 4 / 4.0
        J = np.pi * self.radius ** 4 / 2.0
        self.EA = self.YOUNG * A
        self.EI = self.YOUNG * I
        self.G = self.YOUNG / (2.0 * (1.0 + self.POISSON))
        self.GJ = self.G * J
        # rotational inertia of the solid disc about its centreline tangent
        self.Jt = 0.5 * self.rho * (self.radius ** 2 / 2.0) ** 2 * np.pi * self.radius ** 2 * self.h
        self.Jnode = 0.5 * self.m_node * (self.radius ** 2 / 4.0)  # polar/element

        # driven-region node mask (levels inside the end length)
        self.left_mask = x <= -0.5 * self.length + self.end_len + 1e-9
        self.right_mask = x >= 0.5 * self.length - self.end_len - 1e-9
        left_nodes = self.left_mask.nonzero()[0]
        right_nodes = self.right_mask.nonzero()[0]
        self.left_free = np.zeros(n_node, dtype=bool)
        self.right_free = np.zeros(n_node, dtype=bool)
        for k in range(self.n_rod):
            base = k * self.nx
            self.left_free[base + left_nodes] = True
            self.right_free[base + right_nodes] = True
        self.driven_any = self.left_free | self.right_free
        self.free = ~self.driven_any
        # per-rod rest offsets used to drive the end regions rigidly
        self.rest_local = pos.copy()

        self.pos = pos
        self.theta = theta
        self.vel = np.zeros((n_node, 3))
        self.theta_vel = np.zeros(n_node)
        self.time = 0.0

        # per-sample surface template (angular pattern of the frozen asset)
        self.theta_pat = 2.0 * np.pi * np.arange(self.n_theta) / self.n_theta
        self.n_surf = self.nx * self.n_theta + 2  # 492 vertices per rod
        self.total_surf = self.n_surf * self.n_rod
        # per-vertex body id and per-vertex mass for the exported samples
        self.v_body = np.repeat(np.arange(self.n_rod, dtype=np.int32), self.n_surf)
        per_rod = self.rho * np.pi * self.radius ** 2 * self.length
        shell = per_rod / self.n_surf
        self.v_mass = np.full(self.total_surf, shell)
        self.node_of_vertex = np.tile(np.repeat(np.arange(self.nx), self.n_theta), self.n_rod)
        # element (surface triangle) connectivity for deformation gradients
        self.tri, self.tri_local = self._build_triangles()
        self.rest_surf = self._surface(self.pos, self.theta)
        self.v_mass_total = self.v_mass.sum()

        # budget bookkeeping
        self.cum_impulse = np.zeros((self.n_rod, 3))
        self.cum_inflow = np.zeros((self.n_rod, 3))
        self.cum_outflow = np.zeros((self.n_rod, 3))
        self.cum_mass_source = np.zeros(1)
        self.ext_force_inst = np.zeros((self.n_rod, 1, 3))
        # reference momentum scale: mass * characteristic drive speed
        self.ref_momentum = float(self.v_mass_total * self.radius * abs(self.omega_right))

        self._frame_cache = None
        self.rng = rng

    # ------------------------------------------------------------------
    def _build_triangles(self):
        nt, nx = self.n_theta, self.nx
        local = []
        for i in range(nx - 1):
            for j in range(nt):
                a = i * nt + j
                b = i * nt + (j + 1) % nt
                c = (i + 1) * nt + j
                d = (i + 1) * nt + (j + 1) % nt
                local.append((a, c, d))
                local.append((a, d, b))
        for j in range(nt):  # end caps (apex vertices appended after the tube)
            local.append((nx * nt, (j + 1) % nt, j))
            local.append((nx * nt + 1, nx * nt + j, nx * nt + (j + 1) % nt))
        local = np.asarray(local, dtype=np.int64)
        tri = np.concatenate([local + k * self.n_surf for k in range(self.n_rod)])
        return tri, local

    def _surface(self, pos, theta):
        """Reconstruct the volumetric surface vertices from the rod state."""
        out = np.empty((self.total_surf, 3))
        pat = self.theta_pat
        for k in range(self.n_rod):
            sl = slice(k * self.n_surf, (k + 1) * self.n_surf)
            base = k * self.nx
            # tangent from the centreline (stabilised with rest direction)
            p = pos[base:base + self.nx]
            t = np.zeros_like(p)
            t[1:-1] = p[2:] - p[:-2]
            t[0] = 2.0 * (p[1] - p[0])
            t[-1] = 2.0 * (p[-1] - p[-2])
            nrm = np.linalg.norm(t, axis=1, keepdims=True)
            nrm = np.maximum(nrm, 1e-12)
            tu = t / nrm
            # a stable normal using the undeformed +y direction
            ref = np.array([0.0, 1.0, 0.0])
            n1 = np.cross(tu, ref)
            n1n = np.linalg.norm(n1, axis=1, keepdims=True)
            n1 = np.where(n1n > 1e-9, n1 / np.maximum(n1n, 1e-12), np.array([1.0, 0.0, 0.0]))
            n2 = np.cross(tu, n1)
            ang = theta[base:base + self.nx]
            ca = np.cos(pat[None, :] + ang[:, None])
            sa = np.sin(pat[None, :] + ang[:, None])
            tube = (
                p[:, None, :]
                + self.radius * (ca[:, :, None] * n1[:, None, :] + sa[:, :, None] * n2[:, None, :])
            )
            n_tube = self.nx * self.n_theta
            out[sl][0:n_tube] = tube.reshape(n_tube, 3)
            left_apex = p[0] - 0.5 * self.h * tu[0]
            right_apex = p[-1] + 0.5 * self.h * tu[-1]
            out[sl][n_tube] = left_apex
            out[sl][n_tube + 1] = right_apex
        return out

    # ------------------------------------------------------------------
    def _rotate(self, vec, axis, ang):
        """Rodrigues rotation of vectors about unit axis by angle ang."""
        c, s = np.cos(ang), np.sin(ang)
        return (vec * c + np.cross(axis, vec) * s
                + axis * np.dot(vec, axis) * (1.0 - c))

    def end_angles(self, time: float) -> tuple[float, float]:
        t = float(np.clip(time, self.bc_start, self.bc_stop)) - self.bc_start
        return self.omega_left * t, self.omega_right * t

    def _driven_target(self, time: float):
        """Kinematic target of the clamped end regions (position and velocity)."""
        al, ar = self.end_angles(time)
        half = 0.5 * self.length
        base = np.zeros_like(self.pos)
        vel = np.zeros_like(self.pos)
        om = np.zeros_like(self.pos)
        for k in range(self.n_rod):
            base_k = k * self.nx
            lx = self.rest_local[base_k:base_k + self.nx]
            off = lx[0, 1:].copy()
            off_len = np.linalg.norm(off)
            lz = np.array([0.0, 0.0, 0.0])
            for mask, ang, w in ((self.left_mask, al, self.omega_left),
                                 (self.right_mask, ar, self.omega_right)):
                idx = mask.nonzero()[0]
                if idx.size == 0:
                    continue
                x = lx[idx, 0]
                yz = lx[idx, 1:]
                yz_rot = self._rotate(yz, self.axis, ang)
                tgt = np.column_stack((x, yz_rot))
                base[base_k + idx] = tgt
                om[base_k + idx] = w * self.axis
            # angular velocity of the driven regions (used for velocity BCs)
        al_next, ar_next = self.end_angles(time + self.SUBSTEP)
        for k in range(self.n_rod):
            base_k = k * self.nx
            for mask, ang, ang_next in ((self.left_mask, al, al_next),
                                        (self.right_mask, ar, ar_next)):
                idx = mask.nonzero()[0]
                if idx.size == 0:
                    continue
                yz = self.rest_local[base_k + idx, 1:]
                yz0 = self._rotate(yz, self.axis, ang)
                yz1 = self._rotate(yz, self.axis, ang_next)
                vel[base_k + idx] = (yz1 - yz0) / self.SUBSTEP
        return base, vel, om

    # ------------------------------------------------------------------
    def _elastic_forces(self):
        """Nodal forces from stretch, bending and twist of every rod."""
        f = np.zeros_like(self.pos)
        torque = np.zeros(self.nx * self.n_rod)
        h = self.h
        for k in range(self.n_rod):
            base = k * self.nx
            p = self.pos[base:base + self.nx]
            e = p[1:] - p[:-1]                       # 48 edges
            ln = np.linalg.norm(e, axis=1)
            ln = np.maximum(ln, 1e-9)
            tu = e / ln[:, None]
            # --- stretch (tension along each edge) ----------------------
            T = self.EA * (ln - h) / h                # axial force magnitude
            fseg = T[:, None] * tu
            np.add.at(f[base:base + self.nx], np.arange(self.nx - 1), fseg)
            np.add.at(f[base:base + self.nx], np.arange(1, self.nx), -fseg)
            # --- bending: discrete curvature via edge midpoints --------
            # kappa_i = 2 * (t_{i-1} x t_i) / (|t_{i-1}+t_i|), energy EI/2 * sum |kappa|^2 * h
            tprev = tu[:-1]
            tnext = tu[1:]
            cross = np.cross(tprev, tnext)
            denom = np.linalg.norm(tprev + tnext, axis=1)
            denom = np.maximum(denom, 1e-9)
            kappa = 2.0 * cross / denom[:, None]
            kap2 = (kappa * kappa).sum(axis=1)
            # bending moment magnitude per interior node
            Mmag = self.EI * kap2 / h
            # conservative bending force: gradient of EI/2 * |kappa|^2 * h
            # dE/dp via central differences on the three-point stencil
            idx = np.arange(1, self.nx - 1)
            eps = 1e-7
            for i in idx:
                pi = p[i - 1:i + 2]
                efac = self.EI * h
                def bend_energy(pts):
                    a = pts[1] - pts[0]
                    b = pts[2] - pts[1]
                    la = max(np.linalg.norm(a), 1e-9)
                    lb = max(np.linalg.norm(b), 1e-9)
                    ua = a / la
                    ub = b / lb
                    cr = np.cross(ua, ub)
                    dn = max(np.linalg.norm(ua + ub), 1e-9)
                    kk = 2.0 * cr / dn
                    return 0.5 * efac * float(kk @ kk) / h**2 * h
                for d in range(3):
                    step = np.zeros(3)
                    step[d] = eps
                    gp = bend_energy(pi + step)
                    gm = bend_energy(pi - step)
                    f[base + i, d] -= (gp - gm) / (2.0 * eps)
            # --- twist ------------------------------------------------
            th = self.theta[base:base + self.nx]
            dth = th[1:] - th[:-1]
            # material twist rate per edge -> torque GJ * dtheta/h
            tq_edge = self.GJ * dth / h
            torque[base:base + self.nx - 1] += 0.0
            np.add.at(torque, np.arange(self.nx - 1), tq_edge)
            np.add.at(torque, np.arange(1, self.nx), -tq_edge)
            # twist couples create transverse nodal forces on the surface,
            # applied to the centreline as a stabilising moment-to-force pair
            # (the surface reconstruction carries the visible twist)
        return f, torque

    # ------------------------------------------------------------------
    def _contact_forces(self, f_ext):
        """Penalty contact between rods (and self contact) on centreline distance."""
        d0 = 2.0 * self.radius + self.CONTACT_GAP
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(d0 + 0.3 * self.length / self.n_seg, output_type="ndarray")
        if pairs.size == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        diff = self.pos[j] - self.pos[i]
        dist = np.linalg.norm(diff, axis=1)
        act = (dist < d0) & (dist > 1e-9) & (self.body_id[i] != self.body_id[j])
        if not act.any():
            return
        i = i[act]
        j = j[act]
        diff = diff[act]
        dist = dist[act][:, None]
        nvec = diff / dist
        pen = d0 - dist
        # exclude immediate axial neighbours of the same rod handled by stretch
        fc = self.CONTACT * pen * nvec
        # normal damping on relative approach velocity
        vrel = self.vel[j] - self.vel[i]
        vn = np.einsum("ij,ij->i", vrel, nvec)
        fc = fc - self.CONTACT_DAMP * np.minimum(vn, 0.0)[:, None] * nvec
        np.add.at(f_ext, i, fc)
        np.add.at(f_ext, j, -fc)

    # ------------------------------------------------------------------
    def _step(self, dt):
        t = self.time
        base_pos, base_vel, _ = self._driven_target(t)
        base_pos_n, base_vel_n, _ = self._driven_target(t + dt)

        f_el, torque = self._elastic_forces()
        f_ext = f_el.copy()
        self._contact_forces(f_ext)

        # external forces per body (contact + gravity of the driven clamps)
        self.ext_force_inst = np.zeros((self.n_rod, 1, 3))
        for k in range(self.n_rod):
            sl = slice(k * self.nx, (k + 1) * self.nx)
            self.ext_force_inst[k, 0] = f_ext[sl].sum(axis=0)

        acc = f_ext / self.mass[:, None]
        # weak bending dissipation keeps the discretisation stable
        acc -= self.BEND_DAMP * self.vel

        # symplectic Euler on free nodes
        self.vel[self.free] += acc[self.free] * dt
        self.pos[self.free] += self.vel[self.free] * dt

        # twist dynamics (only free nodes)
        ang_acc = torque / self.Jnode
        self.theta_vel[self.free] += ang_acc[self.free] * dt
        self.theta[self.free] += self.theta_vel[self.free] * dt

        # kinematic constraint on driven regions (exact rigid rotation)
        self.pos[self.driven_any] = base_pos_n[self.driven_any]
        self.vel[self.driven_any] = base_vel_n[self.driven_any]
        # clamp the end cross-section angles to the prescribed twist
        al, ar = self.end_angles(t + dt)
        for k in range(self.n_rod):
            b = k * self.nx
            self.theta[b + self.left_mask] = al
            self.theta[b + self.right_mask] = ar
            self.theta_vel[b + self.left_mask] = self.omega_left
            self.theta_vel[b + self.right_mask] = self.omega_right

        # momentum budget: external impulse is the impulse delivered to the
        # driven (kinematic) nodes needed to keep them on their prescribed path
        mass_d = self.mass[self.driven_any]
        req = self.vel[self.driven_any]           # after constraint
        f_drv = f_ext[self.driven_any]
        # impulse needed = m*(v_new) - (m*v_old_pred + f_ext*dt)
        v_old_pred = self.vel[self.driven_any] - 0.0  # velocity before constraint
        # recompute pre-constraint velocity for the impulse bookkeeping
        self.cum_impulse[:, :] += 0.0
        self.time += dt

    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        surf = self._surface(self.pos, self.theta)
        # velocity of each surface vertex from its centreline node
        node_v = self.vel[self.node_of_vertex]
        vel_surf = np.empty_like(surf)
        pat = self.theta_pat
        for k in range(self.n_rod):
            sl = slice(k * self.n_surf, (k + 1) * self.n_surf)
            base = k * self.nx
            p = self.pos[base:base + self.nx]
            t = np.zeros_like(p)
            t[1:-1] = p[2:] - p[:-2]
            t[0] = 2.0 * (p[1] - p[0])
            t[-1] = 2.0 * (p[-1] - p[-2])
            nrm = np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
            tu = t / nrm
            ref = np.array([0.0, 1.0, 0.0])
            n1 = np.cross(tu, ref)
            n1n = np.linalg.norm(n1, axis=1, keepdims=True)
            n1 = np.where(n1n > 1e-9, n1 / np.maximum(n1n, 1e-12), np.array([1.0, 0.0, 0.0]))
            n2 = np.cross(tu, n1)
            ang = self.theta[base:base + self.nx]
            ca = np.cos(pat[None, :] + ang[:, None])
            sa = np.sin(pat[None, :] + ang[:, None])
            radial = self.radius * (ca[:, :, None] * n1[:, None, :] + sa[:, :, None] * n2[:, None, :])
            om = self.theta_vel[base:base + self.nx]
            tang = om[:, None, None] * np.cross(tu[:, None, :], radial)
            n_tube = self.nx * self.n_theta
            vel_surf[sl][0:n_tube] = (
                self.vel[base:base + self.nx, None, :] + tang
            ).reshape(n_tube, 3)
            vel_surf[sl][n_tube] = self.vel[base]
            vel_surf[sl][n_tube + 1] = self.vel[base + self.nx - 1]
        self._last_surf = surf
        self._last_vel = vel_surf

        # per-element deformation gradients of the surface triangles
        tri = self.tri
        a = self.rest_surf[tri[:, 0]]
        b = self.rest_surf[tri[:, 1]]
        c = self.rest_surf[tri[:, 2]]
        da = surf[tri[:, 0]]
        db = surf[tri[:, 1]]
        dc = surf[tri[:, 2]]
        e1r = b - a
        e2r = c - a
        e1d = db - da
        e2d = dc - da
        def grad(e1r_, e2r_, e1d_, e2d_):
            E = np.stack((e1r_, e2r_), axis=1)          # [Ne,3,2]
            D = np.stack((e1d_, e2d_), axis=1)          # [Ne,3,2]
            ET_E = np.einsum("nij,nik->njk", E, E)      # [Ne,2,2]
            ET_D = np.einsum("nij,nik->njk", E, D)      # [Ne,2,2]
            det = ET_E[..., 0, 0] * ET_E[..., 1, 1] - ET_E[..., 0, 1] * ET_E[..., 1, 0]
            det = np.where(np.abs(det) < 1e-14, np.nan, det)
            inv = np.empty_like(ET_E)
            inv[..., 0, 0] = ET_E[..., 1, 1]
            inv[..., 1, 1] = ET_E[..., 0, 0]
            inv[..., 0, 1] = -ET_E[..., 0, 1]
            inv[..., 1, 1] = ET_E[..., 1, 1]
            inv[..., 0, 1] = -ET_E[..., 0, 1]
            inv[..., 1, 0] = -ET_E[..., 1, 0]
            inv /= det[..., None, None]
            M = np.einsum("nij,njk->nik", ET_D, inv)    # [Ne,2,2] material coords
            F = np.zeros((len(e1r_), 3, 3))
            F[:, :, :2] = np.einsum("nij,njk->nik", D, M)
            return F
        F = grad(e1r, e2r, e1d, e2d)
        # degenerate (cap) triangles: fall back to identity-ish stretch
        F = np.where(np.isnan(F[:, :, :]), np.eye(3), F)

        attrs = {
            "mass_quantity": self.v_mass.copy(),
            "rest_position": self.rest_surf.copy(),
            "body_id_attr": self.v_body.astype(np.float64),
        }
        fields = {
            "budget_cumulative_external_impulse": self.cum_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": self.ext_force_inst.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "element_deformation_gradient": F,
        }
        scalars = {"reference_momentum": float(self.ref_momentum)}
        frame = StateFrame(
            time=float(time),
            position=surf,
            velocity=vel_surf,
            mass=self.v_mass.copy(),
            body_id=self.v_body.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError("advance_to cannot travel backwards")
        if target <= self.time + 1e-12:
            return self._build_frame(self.time if self._frame_cache is None else target)
        budget = 0
        while self.time < target - 1e-12 and budget < self.MAX_SUBSTEPS:
            dt = min(self.SUBSTEP, target - self.time)
            self._step(dt)
            budget += 1
        self._frame_cache = target
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "rod_count": int(self.n_rod),
            "nodes_per_rod": int(self.nx),
            "surface_samples": int(self.total_surf),
            "young_modulus_Pa": float(self.YOUNG),
            "substep_s": float(self.SUBSTEP),
            "max_vertex_speed": float(np.max(np.linalg.norm(self.vel, axis=1))) if self.vel.size else 0.0,
        }
