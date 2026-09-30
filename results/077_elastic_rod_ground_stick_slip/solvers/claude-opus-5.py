"""Corotational FEM solver for stick-slip of an elastic rod dragged on a plane.

Physical model
--------------
The rod is a thin volumetric hexagonal-prism solid discretised into linear
tetrahedra built from the frozen surface samples of ``scene.py`` (60 vertices,
10 rings x 6 sides).  Elasticity is corotational linear FEM (polar
decomposition of the per-element deformation gradient), integrated with a
linearised backward-Euler (warped-stiffness) scheme, so the stiff soft-solid
dynamics are unconditionally stable at a fixed step.

Contact with the frozen ground box (top face at z = 0) is treated as a
unilateral velocity-level constraint with Baumgarte penetration recovery; the
bottom cap samples carry the rod contact thickness (capsule cap of radius
``rod_radius``) so the rod touches the plane at t = 0 as required.  Dry
friction is a per-contact Coulomb solve: the tangential impulse that would stop
the sliding is clamped by mu * N, which yields genuine stick and slip states
and hence the stick / bend / slip limit cycle.

All reported forces and impulses are the discrete quantities actually used in
the update, so the momentum budget closes to round-off.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


def _hex_prism_tets(rings: int, sides: int) -> np.ndarray:
    """Tetrahedralise the hexagonal prism shell using only surface samples."""
    fan = [(0, 2, 3), (0, 1, 2), (0, 3, 4), (0, 4, 5)]
    tets: list[tuple[int, int, int, int]] = []
    for k in range(rings - 1):
        base = k * sides
        top = (k + 1) * sides
        for (a, b, c) in fan:
            p = (base + a, base + b, base + c)
            q = (top + a, top + b, top + c)
            tets.append((p[0], p[1], p[2], q[2]))
            tets.append((p[0], p[1], q[2], q[1]))
            tets.append((p[0], q[1], q[2], q[0]))
    return np.asarray(tets, dtype=np.int64)


def _elastic_matrix(young: float, poisson: float) -> np.ndarray:
    """Isotropic linear-elastic Voigt matrix (engineering shear strains)."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    c = np.zeros((6, 6), dtype=float)
    c[:3, :3] = lam
    c[0, 0] = c[1, 1] = c[2, 2] = lam + 2.0 * mu
    c[3, 3] = c[4, 4] = c[5, 5] = mu
    return c


def _smoothstep(u: np.ndarray | float) -> float:
    u = float(min(max(u, 0.0), 1.0))
    return u * u * (3.0 - 2.0 * u)


def _smoothstep_deriv(u: float) -> float:
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 6.0 * u * (1.0 - u)


class BenchmarkSolver:
    """Corotational-FEM rod with unilateral Coulomb ground contact."""

    # material and numerical constants (frozen at initialize time)
    YOUNG = 1.0e5          # Pa, soft rubbery rod -> bending mode near 1.7 Hz
    POISSON = 0.40
    RAYLEIGH_BETA = 4.0e-3  # stiffness-proportional damping (internal, sums to 0)
    MAX_SUBSTEP = 4.0e-4    # s, backward-Euler substep
    IMPERFECTION = 2.0e-3   # m, rest-shape bow that selects the buckling plane
    STICK_TOL = 1.0e-3      # m/s, stick/slip classification tolerance
    CONTACT_BAND = 1.0e-3   # m, gap band used by the frozen contact selection

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        self.config = cfg
        frame0 = scene.initial_state()

        self.x0 = np.asarray(frame0.position, dtype=float).copy()
        self.n = int(self.x0.shape[0])
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.total_mass = float(self.mass.sum())
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        attrs = {k: np.asarray(v, dtype=float) for k, v in frame0.attributes.items()}
        self.driven_mask = attrs["driven_top"] > 0.5
        self.driven_idx = np.flatnonzero(self.driven_mask)
        self.free_idx = np.flatnonzero(~self.driven_mask)

        self.rod_radius = float(frame0.scalars.get("rod_radius", 0.055))
        self.rod_height = float(frame0.scalars.get("rod_height", 1.45))

        # ---- mesh topology (hexagonal prism shell -> tetrahedra) -----------
        z_sorted = np.unique(np.round(self.x0[:, 2], 9))
        self.rings = int(z_sorted.size)
        self.sides = self.n // self.rings
        order = np.lexsort((np.arange(self.n), self.x0[:, 2]))
        if not np.array_equal(order, np.arange(self.n)):
            raise ValueError("unexpected vertex ordering in scene samples")
        self.tets = _hex_prism_tets(self.rings, self.sides)

        # ---- rest configuration with a small in-plane bow (imperfection) ---
        rest = self.x0.copy()
        z_lo = float(self.x0[:, 2].min())
        span = max(float(self.x0[:, 2].max()) - z_lo, 1e-9)
        s = (self.x0[:, 2] - z_lo) / span
        rest[:, 0] += self.IMPERFECTION * np.sin(np.pi * s)
        self.rest = rest

        self._build_fem()

        # ---- contact model -------------------------------------------------
        self.ground_top = 0.0
        self.contact_offset = np.zeros(self.n, dtype=float)
        bottom = self.x0[:, 2] <= z_lo + 1e-9
        self.contact_offset[bottom] = self.rod_radius
        self.mu = float(cfg["contact"]["ground_friction_coefficient"])
        self.n_contacts = self.n
        self.ground_normal = np.zeros((self.n_contacts, 3), dtype=float)
        self.ground_normal[:, 2] = 1.0

        # ---- prescribed top motion ----------------------------------------
        bc = cfg["boundary_conditions"]["driven_top"]
        self.press_t0, self.press_t1 = (float(v) for v in bc["press_interval"])
        self.press_disp = np.asarray(bc["press_displacement"], dtype=float)
        self.drag_t0, self.drag_t1 = (float(v) for v in bc["drag_interval"])
        self.drag_vel = np.asarray(bc["drag_velocity"], dtype=float)
        self.drag_ramp = 0.06

        # ---- integration state --------------------------------------------
        self.time = 0.0
        self.pos = self.x0.copy()
        self.vel = np.zeros_like(self.x0)
        self.cum_impulse = np.zeros((4, 3), dtype=float)
        self.last_force = np.zeros((4, 3), dtype=float)
        self._contact = self._empty_contact()
        self._nsteps = 0

    # ------------------------------------------------------------- FEM setup
    def _build_fem(self) -> None:
        tets = self.tets
        p = self.rest[tets]                       # [Ne,4,3]
        edges = np.stack(
            (p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2
        )                                          # [Ne,3,3] columns are edges
        det = np.linalg.det(edges)
        flip = det < 0.0
        if np.any(flip):
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            self.tets = tets
            p = self.rest[tets]
            edges = np.stack(
                (p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2
            )
            det = np.linalg.det(edges)
        self.ne = int(tets.shape[0])
        self.edge_inv = np.linalg.inv(edges)       # [Ne,3,3], F = E_cur @ edge_inv
        self.vol = det / 6.0
        self.rest_volume = float(self.vol.sum())

        # shape-function gradients: rows of edge_inv give nodes 1..3
        grad = np.zeros((self.ne, 4, 3), dtype=float)
        grad[:, 1:, :] = np.transpose(self.edge_inv, (0, 2, 1))
        grad[:, 0, :] = -grad[:, 1:, :].sum(axis=1)
        self.grad = grad

        cmat = _elastic_matrix(self.YOUNG, self.POISSON)
        bmat = np.zeros((self.ne, 6, 12), dtype=float)
        for a in range(4):
            g = grad[:, a, :]
            bmat[:, 0, 3 * a + 0] = g[:, 0]
            bmat[:, 1, 3 * a + 1] = g[:, 1]
            bmat[:, 2, 3 * a + 2] = g[:, 2]
            bmat[:, 3, 3 * a + 0] = g[:, 1]
            bmat[:, 3, 3 * a + 1] = g[:, 0]
            bmat[:, 4, 3 * a + 1] = g[:, 2]
            bmat[:, 4, 3 * a + 2] = g[:, 1]
            bmat[:, 5, 3 * a + 0] = g[:, 2]
            bmat[:, 5, 3 * a + 2] = g[:, 0]
        self.k_elem = self.vol[:, None, None] * np.einsum(
            "eij,jk,ekl->eil", np.transpose(bmat, (0, 2, 1)), cmat, bmat
        )
        self.lam = self.YOUNG * self.POISSON / ((1.0 + self.POISSON) * (1.0 - 2.0 * self.POISSON))
        self.mu_lame = self.YOUNG / (2.0 * (1.0 + self.POISSON))

        dofs = (3 * tets[:, :, None] + np.arange(3)[None, None, :]).reshape(self.ne, 12)
        self.rows = np.repeat(dofs, 12, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 12)).ravel()
        self.ndof = 3 * self.n
        self.m_diag = np.repeat(self.mass, 3)
        self.f_grav = self.mass[:, None] * self.gravity[None, :]

    # --------------------------------------------------- prescribed top motion
    def _top_velocity(self, t: float) -> np.ndarray:
        v = np.zeros(3, dtype=float)
        dur = max(self.press_t1 - self.press_t0, 1e-12)
        if self.press_t0 <= t < self.press_t1:
            u = (t - self.press_t0) / dur
            v += self.press_disp * (_smoothstep_deriv(u) / dur)
        if self.drag_t0 <= t <= self.drag_t1:
            ramp = min(max((t - self.drag_t0) / self.drag_ramp, 0.0), 1.0)
            v += self.drag_vel * _smoothstep(ramp)
        return v

    # ------------------------------------------------------- internal forces
    def _internal(self, pos: np.ndarray):
        p = pos[self.tets]
        edges = np.stack(
            (p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2
        )
        fdef = edges @ self.edge_inv                     # [Ne,3,3]
        u, _s, vt = np.linalg.svd(fdef)
        rot = u @ vt
        neg = np.linalg.det(rot) < 0.0
        if np.any(neg):
            u[neg, :, 2] *= -1.0
            rot = u @ vt
        # corotational linear strain and stress
        sym = 0.5 * (np.einsum("eji,ejk->eik", rot, fdef) + np.einsum("eji,ejk->eik", fdef, rot))
        eps = sym - np.eye(3)[None, :, :]
        tr = np.trace(eps, axis1=1, axis2=2)
        stress = 2.0 * self.mu_lame * eps + self.lam * tr[:, None, None] * np.eye(3)[None, :, :]
        pk = rot @ stress                                 # rotated stress
        # nodal forces: f_a = -V * P @ grad_a
        fel = -self.vol[:, None, None] * np.einsum("eij,eaj->eai", pk, self.grad)
        fint = np.zeros((self.n, 3), dtype=float)
        np.add.at(fint, self.tets.ravel(), fel.reshape(-1, 3))
        return fint, rot, fdef

    def _stiffness(self, rot: np.ndarray) -> np.ndarray:
        rb = np.zeros((self.ne, 12, 12), dtype=float)
        for a in range(4):
            rb[:, 3 * a:3 * a + 3, 3 * a:3 * a + 3] = rot
        kw = rb @ self.k_elem @ np.transpose(rb, (0, 2, 1))
        kmat = np.zeros((self.ndof, self.ndof), dtype=float)
        np.add.at(kmat, (self.rows, self.cols), kw.ravel())
        return kmat

    @staticmethod
    def _solve_constrained(amat, bvec, fixed, fixed_val, extra):
        """Solve A v = b + extra with v[fixed] = fixed_val; return v, reaction."""
        rhs = bvec + extra
        free = ~fixed
        vsol = np.zeros_like(rhs)
        vsol[fixed] = fixed_val[fixed]
        rr = rhs[free] - amat[np.ix_(free, fixed)] @ vsol[fixed]
        vsol[free] = np.linalg.solve(amat[np.ix_(free, free)], rr)
        react = amat[np.ix_(fixed, np.arange(len(rhs)))] @ vsol - rhs[fixed]
        return vsol, react

    # ------------------------------------------------------------- one substep
    def _step(self, dt: float) -> None:
        pos, vel = self.pos, self.vel
        fint, rot, _f = self._internal(pos)
        kmat = self._stiffness(rot)
        amat = np.diag(self.m_diag) + (dt * self.RAYLEIGH_BETA + dt * dt) * kmat

        f_grav = self.f_grav
        rhs = self.m_diag * vel.ravel() + dt * (fint + f_grav).ravel()

        # prescribed top-grip velocity
        v_top = self._top_velocity(self.time + dt)
        fixed_val = np.zeros(self.ndof, dtype=float)
        base_fixed = np.zeros(self.ndof, dtype=bool)
        for i in self.driven_idx:
            base_fixed[3 * i:3 * i + 3] = True
            fixed_val[3 * i:3 * i + 3] = v_top

        # contact candidate set (frozen rule: bottom cap samples near the plane)
        gap = pos[:, 2] - self.contact_offset - self.ground_top
        cand = (self.contact_offset > 0.0) & (~self.driven_mask) & (gap < self.CONTACT_BAND)
        cand_idx = np.flatnonzero(cand)
        active = np.ones(cand_idx.size, dtype=bool)
        sticking = np.ones(cand_idx.size, dtype=bool)
        vt_dir = np.zeros((cand_idx.size, 3), dtype=float)
        if cand_idx.size:
            vt_prev = vel[cand_idx].copy()
            vt_prev[:, 2] = 0.0
            speed = np.linalg.norm(vt_prev, axis=1)
            moving = speed > self.STICK_TOL
            vt_dir[moving] = vt_prev[moving] / speed[moving, None]
            sticking = ~moving

        fn = np.zeros(cand_idx.size, dtype=float)
        ft = np.zeros((cand_idx.size, 3), dtype=float)
        vnew = vel.ravel().copy()
        react_top = np.zeros(3, dtype=float)

        for _it in range(4):
            fixed = base_fixed.copy()
            fval = fixed_val.copy()
            extra = np.zeros(self.ndof, dtype=float)
            for c, i in enumerate(cand_idx):
                if not active[c]:
                    continue
                g = gap[i]
                fixed[3 * i + 2] = True
                fval[3 * i + 2] = max(-g, 0.0) * 0.3 / dt
                if sticking[c]:
                    fixed[3 * i] = True
                    fixed[3 * i + 1] = True
                    fval[3 * i] = 0.0
                    fval[3 * i + 1] = 0.0
                else:
                    extra[3 * i:3 * i + 3] += dt * ft[c]
            vsol, react = self._solve_constrained(amat, rhs, fixed, fval, extra)
            fixed_pos = np.flatnonzero(fixed)
            lam = np.zeros(self.ndof, dtype=float)
            lam[fixed_pos] = react / dt
            changed = self._update_contact_state(
                cand_idx, active, sticking, vt_dir, fn, ft, lam, vsol, dt
            )
            vnew = vsol
            react_top = np.array(
                [lam[3 * i:3 * i + 3] for i in self.driven_idx], dtype=float
            ).sum(axis=0)
            if not changed:
                break

        # record contact diagnostics for the frame exports
        self._store_contact(cand_idx, active, sticking, fn, ft, vnew.reshape(self.n, 3))
        self.vel = vnew.reshape(self.n, 3)
        self.pos = pos + dt * self.vel
        # cumulative external impulses (gravity, contact normal, friction, grip)
        self.cum_impulse[0] += dt * f_grav.sum(axis=0)
        f_normal = np.zeros(3, dtype=float)
        f_fric = np.zeros(3, dtype=float)
        for c in range(cand_idx.size):
            if not active[c]:
                continue
            f_normal[2] += fn[c]
            f_fric += ft[c]
        self.cum_impulse[1] += dt * f_normal
        self.cum_impulse[2] += dt * f_fric
        self.cum_impulse[3] += dt * react_top
        self.last_force = np.stack((f_grav.sum(axis=0), f_normal, f_fric, react_top))
        self.time += dt
        self._nsteps += 1

    def _update_contact_state(self, cand_idx, active, sticking, vt_dir, fn, ft, lam, vsol, dt):
        changed = False
        v3 = vsol.reshape(self.n, 3)
        for c, i in enumerate(cand_idx):
            if not active[c]:
                continue
            nf = lam[3 * i + 2]
            if nf <= 0.0:
                active[c] = False
                fn[c] = 0.0
                ft[c] = 0.0
                changed = True
                continue
            fn[c] = nf
            limit = self.mu * nf
            if sticking[c]:
                tang = np.array([lam[3 * i], lam[3 * i + 1], 0.0])
                mag = float(np.linalg.norm(tang))
                if mag > limit and mag > 1e-12:
                    sticking[c] = False
                    d = tang / mag
                    vt_dir[c] = -d
                    ft[c] = limit * d
                    changed = True
                else:
                    ft[c] = tang
            else:
                vt = v3[i].copy()
                vt[2] = 0.0
                sp = float(np.linalg.norm(vt))
                if sp <= self.STICK_TOL:
                    sticking[c] = True
                    ft[c] = 0.0
                    changed = True
                else:
                    d = vt / sp
                    if float(d @ vt_dir[c]) < 0.99:
                        vt_dir[c] = d
                        changed = True
                    ft[c] = -limit * vt_dir[c]
        return changed

    # ------------------------------------------------------- contact bookkeeping
    def _empty_contact(self) -> dict:
        c = self.n_contacts
        return {
            "mask": np.zeros(c, dtype=float),
            "normal": self.ground_normal.copy(),
            "fn": np.zeros(c, dtype=float),
            "ft": np.zeros((c, 3), dtype=float),
            "vrel": np.zeros((c, 3), dtype=float),
        }

    def _store_contact(self, cand_idx, active, sticking, fn, ft, vnew) -> None:
        rec = self._empty_contact()
        for c, i in enumerate(cand_idx):
            if not active[c]:
                continue
            rec["mask"][i] = 1.0
            rec["fn"][i] = max(float(fn[c]), 0.0)
            rec["ft"][i] = ft[c]
            v = vnew[i].copy()
            v[2] = 0.0 if sticking[c] else v[2]
            rec["vrel"][i] = vnew[i]
        self._contact = rec

    # --------------------------------------------------------------- advancing
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.MAX_SUBSTEP, target - self.time)
            self._step(dt)
        self.time = target
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        pos = np.ascontiguousarray(self.pos, dtype=float)
        vel = np.ascontiguousarray(self.vel, dtype=float)
        rec = self._contact
        c = self.n_contacts

        # per-element deformation gradients
        p = pos[self.tets]
        edges = np.stack(
            (p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2
        )
        fdef = np.ascontiguousarray(edges @ self.edge_inv)

        # ---- momentum budget: one global domain (G = 1) -------------------
        weights = np.ones((1, self.n), dtype=float)
        ext_force = self.last_force.reshape(1, 4, 3).copy()
        cum_imp = self.cum_impulse.sum(axis=0).reshape(1, 3).copy()
        zero_g = np.zeros((1, 3), dtype=float)

        vref = max(float(np.linalg.norm(self.drag_vel)), 1e-3)
        fref = max(self.total_mass * float(np.linalg.norm(self.gravity)), 1e-6)

        fields = {
            "budget_domain_weights": weights,
            "budget_external_force": ext_force,
            "budget_cumulative_external_impulse": cum_imp,
            "budget_cumulative_inflow_momentum": zero_g,
            "budget_cumulative_outflow_momentum": zero_g,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "element_deformation_gradient": fdef,
            "friction_active_mask": rec["mask"],
            "friction_normal": rec["normal"],
            "friction_normal_force": rec["fn"],
            "friction_tangential_force": rec["ft"],
            "friction_relative_velocity": rec["vrel"],
            "friction_static_coefficient": np.full(c, self.mu, dtype=float),
            "friction_dynamic_coefficient": np.full(c, self.mu, dtype=float),
        }
        scalars = {
            "friction_stick_speed_tolerance": float(self.STICK_TOL),
            "reference_force": float(fref),
            "reference_momentum": float(self.total_mass * vref),
            "reference_velocity": float(vref),
            "rod_radius": self.rod_radius,
            "rod_height": self.rod_height,
            "ground_friction_coefficient": self.mu,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "driven_top": self.driven_mask.astype(float),
            "rod_vertex": np.ones(self.n, dtype=float),
        }
        frame = StateFrame(
            max(float(t), 0.0),
            pos,
            vel,
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "vertices": self.n,
            "elements": self.ne,
            "substeps": self._nsteps,
            "substep_dt": self.MAX_SUBSTEP,
            "rest_volume": self.rest_volume,
            "young_modulus": self.YOUNG,
            "friction_coefficient": self.mu,
        }
