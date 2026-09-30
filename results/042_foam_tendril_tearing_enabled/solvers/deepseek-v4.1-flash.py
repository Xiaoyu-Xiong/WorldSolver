"""Deterministic cohesive-elastic-plastic foam pull-apart solver (Figure 7).

Physical model
--------------
The shaving-foam mass is discretised as a Lagrangian particle continuum: every
material sample carries a mass and every initial bond between neighbouring
samples carries a rate-dependent elastic-plastic-damage axial law

    elastic   : f = k(s) * s,              s = L / L0e - 1
    visco     : f += c * (v_rel . n)
    plastic   : dL0e = max(0, s - s_y),    (irreversible flow, softening)
    damage    : dD   = alpha * dL0e + beta * max(s, 0) dt,   rupture at D >= 1

Tension yields and damages the thin bridge first because the frozen geometry
gives the neck a small initial section (large flaw density), so the deformation
localises there: the tendril thins, ruptures explicitly, and no bond is ever
re-created, which removes the nonphysical persistent thread.  Stress free
remnants retract and settle as two disconnected fragments.

Boundary conditions: gravity, a penalty/friction contact with the matte floor
cylinder, a penalty contact with the dark upper plate, and adhesion of the top
layer to the plate, which separates at the prescribed pull velocity.

The deformation gradient is recovered per sample with a moving-least-squares
fit of the current bond vectors against the reference bond vectors.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# Frozen scaffold geometry (scene_geometry.yaml, metres).
_FLOOR_TOP = 0.0
_FLOOR_RADIUS = 1.35
_PLATE_HX = 0.78
_PLATE_HY = 0.62
_PLATE_BOTTOM0 = 1.84  # box centre 1.92 minus half extent 0.08

# Material / numerical parameters of the foam lattice.
_YOUNG = 6000.0          # Pa, soft shaving foam
_ZETA = 0.5              # bond dashpot critical damping ratio
_WALL_K = 1.2e3          # N/m soft penalty stiffness (floor and plate)
_WALL_ZETA = 0.7
_FRICTION = 0.7
_ADHESION_K = 3.0e3      # N/m, top layer stuck to the plate
_ADHESION_ZETA = 0.7
_AIR_DRAG = 0.6          # 1/s, viscous dissipation of the foam
_PLASTIC_RATE = 30.0     # 1/s, max viscoplastic flow rate
_ALPHA = 3.8             # damage per unit plastic strain (rupture at 1/alpha)
_CREEP = 1.2             # 1/s, time dependent damage under sustained tension
_S_YIELD_MIN = 0.05
_S_YIELD_MAX = 0.32
_MAX_SPEED = 6.0
_DT = 4.0e-4
_SAMPLE_TOL = 1e-9


class BenchmarkSolver:
    """Baseline solver implementing the Scene/Solver contract."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = scene.config
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=np.float64)
        self.v = np.array(frame0.velocity, dtype=np.float64)
        self.m = np.array(frame0.mass, dtype=np.float64)
        self.n = int(self.x.shape[0])
        if self.m.shape != (self.n,):
            self.m = np.full(self.n, float(np.mean(self.m)))
        self.reference = self.x.copy()

        initial = config["initial_conditions"]
        self.spacing = float(initial["particle_spacing"])
        self.radius = float(initial["nominal_particle_radius"])
        self.density = float(initial["bulk_density"])
        self.pull = float(initial["upper_pull_velocity"][2])
        self.gvec = np.array(config["gravity"], dtype=np.float64)
        self.dt = _DT
        self.stroke = 0.16  # commanded separation of the upper support (m)

        radius_attr = frame0.attributes.get("particle_radius")
        self.rad = (
            np.asarray(radius_attr, dtype=np.float64).copy()
            if radius_attr is not None
            else np.full(self.n, self.radius)
        )
        tear_attr = frame0.attributes.get("tear_candidate")
        self.tear_candidate = (
            np.asarray(tear_attr, dtype=np.float64).copy()
            if tear_attr is not None
            else np.zeros(self.n)
        )

        self._build_bonds()
        self._build_flaw_field()

        self.plastic_rate = _PLASTIC_RATE
        self.alpha = _ALPHA
        self.creep = _CREEP
        self.kwall = _WALL_K
        self.cwall = 2.0 * _WALL_ZETA * np.sqrt(_WALL_K * float(np.mean(self.m)))
        self.ctan = 0.5 * self.cwall
        self.kadh = _ADHESION_K
        self.cadh = 2.0 * _ADHESION_ZETA * np.sqrt(_ADHESION_K * float(np.mean(self.m)))

        self._time = 0.0
        self.steps = 0
        self.ruptured_total = 0

    def _build_bonds(self) -> None:
        x = self.x
        tree = cKDTree(x)
        pairs = tree.query_pairs(1.55 * self.spacing, output_type="ndarray")
        if pairs.size == 0:
            pairs = tree.query(x, k=7)[1][:, 1:].reshape(-1, 2)[:, ::-1]
            pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        self.pi = np.ascontiguousarray(pairs[:, 0])
        self.pj = np.ascontiguousarray(pairs[:, 1])
        self.nbonds = int(self.pi.size)

        ref = self.reference[self.pi] - self.reference[self.pj]
        self.L0 = np.maximum(np.linalg.norm(ref, axis=1), 1e-6)
        self.L0e = self.L0.copy()
        self.D = np.zeros(self.nbonds, dtype=np.float64)
        self.active = np.ones(self.nbonds, dtype=bool)

        mass_bar = float(np.mean(self.m))
        kt = _YOUNG * self.spacing**2 / self.L0
        self.kt = kt
        self.kc = 3.0 * kt
        self.cb = 2.0 * _ZETA * np.sqrt(kt * mass_bar)

    def _build_flaw_field(self) -> None:
        """Frozen structural weakness from the initial thickness profile.

        Thin sections contain relatively larger flaws, so they yield first.
        The profile is measured from the frozen initial particle cloud only.
        """
        x = self.reference
        rxy = np.hypot(x[:, 0], x[:, 1])
        z = x[:, 2]
        slab = 0.07
        bins = np.floor(z / slab).astype(np.int64)
        nb = int(bins.max()) + 1
        thick = np.zeros(nb)
        for b in range(nb):
            sel = bins == b
            thick[b] = np.percentile(rxy[sel], 97.0) if np.any(sel) else 0.0
        # smooth the profile a little to avoid a jagged strength field
        kern = np.array([0.25, 0.5, 0.25])
        thick = np.convolve(thick, kern, mode="same")
        local = thick[np.clip(bins, 0, nb - 1)]
        ref = max(float(np.percentile(local, 92.0)), 1e-6)
        norm = np.clip(local / ref, 0.10, 1.0)

        s_y = _S_YIELD_MIN + (_S_YIELD_MAX - _S_YIELD_MIN) * norm
        # gentle extra weakening inside the intended tear band
        zc = 0.5 * (0.78 + 1.12)
        half = 0.5 * (1.12 - 0.78)
        bump = np.clip(1.0 - np.abs(z - zc) / (half + 0.12), 0.0, 1.0)
        s_y *= 1.0 - 0.25 * bump
        self.sy_particle = s_y
        self.sy = 0.5 * (s_y[self.pi] + s_y[self.pj])

        # particles stuck to the plate underside
        self.adh = np.flatnonzero(self.reference[:, 2] >= _PLATE_BOTTOM0 - 3.0 * self.radius)

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        while self._time < target - _SAMPLE_TOL:
            h = min(self.dt, target - self._time)
            self._step(h)
        return self._frame(target)

    def diagnostics(self) -> dict:
        return {
            "particle_count": int(self.n),
            "bond_count": int(self.nbonds),
            "steps": int(self.steps),
            "broken_bonds": int(np.count_nonzero(~self.active)),
            "simulation_time": float(self._time),
            "solver": "cohesive_lattice_elastoplastic_damage",
            "explicit_tearing": True,
        }

    # ----------------------------------------------------------------- physics
    def _plate_disp(self, t: float) -> float:
        return float(min(self.pull * t, self.stroke))

    def _step(self, h: float) -> None:
        x, v = self.x, self.v
        pi, pj = self.pi, self.pj

        dx = x[pi] - x[pj]
        L = np.sqrt((dx * dx).sum(1)) + 1e-12
        n = dx / L[:, None]
        s = L / self.L0e - 1.0

        tension = self.active & (s > 0.0)
        if tension.any():
            idx = np.flatnonzero(tension)
            excess = s[idx] - self.sy[idx]
            flow = np.clip(excess, 0.0, self.plastic_rate * h)
            self.L0e[idx] *= 1.0 + flow
            self.D[idx] += self.alpha * flow + self.creep * np.maximum(s[idx], 0.0) * h
            self.ruptured_total += int(np.count_nonzero(self.D[idx] >= 1.0))

        dead = self.active & ((self.D >= 1.0) | (s > 1.6))
        if dead.any():
            self.active[dead] = False

        stiff = np.where(s > 0.0, self.kt * (1.0 - self.D), self.kc)
        damp = self.cb * (1.0 - np.minimum(self.D, 0.99))
        vn = ((v[pi] - v[pj]) * n).sum(1)
        fmag = (stiff * s + damp * vn) * self.active

        force = np.zeros_like(x)
        for a in range(3):
            col = -fmag * n[:, a]
            force[:, a] += np.bincount(pi, weights=col, minlength=self.n)
            force[:, a] += np.bincount(pj, weights=-col, minlength=self.n)

        force += self.m[:, None] * self.gvec[None, :]
        force -= self.m[:, None] * _AIR_DRAG * v

        t0 = self._time
        disp = self._plate_disp(t0)
        rising = self.pull * t0 < self.stroke
        self._boundary_forces(force, disp, rising)

        v += (force / self.m[:, None]) * h
        speed = np.linalg.norm(v, axis=1)
        fast = speed > _MAX_SPEED
        if fast.any():
            v[fast] *= _MAX_SPEED / speed[fast][:, None]
        x += v * h
        self._time = t0 + h
        self.steps += 1

    def _boundary_forces(self, force, disp: float, rising: bool) -> None:
        x, v = self.x, self.v
        rad = self.rad

        # --- floor: top face of the cylinder at z = 0, radius 1.35 -----------
        pen = (_FLOOR_TOP + rad) - x[:, 2]
        on = pen > 0.0
        if on.any():
            fn = np.maximum(self.kwall * pen[on] - self.cwall * v[on, 2], 0.0)
            force[on, 2] += fn
            vt = v[on, :2]
            sp = np.linalg.norm(vt, axis=1)
            ft = np.minimum(self.ctan * sp, _FRICTION * fn)
            scale = np.where(sp > 1e-9, ft / np.maximum(sp, 1e-9), 0.0)
            force[on, 0] -= scale * vt[:, 0]
            force[on, 1] -= scale * vt[:, 1]

        # --- upper plate: box underside at z = 1.84 + disp --------------------
        zplate = _PLATE_BOTTOM0 + disp
        inb = (np.abs(x[:, 0]) <= _PLATE_HX) & (np.abs(x[:, 1]) <= _PLATE_HY)
        penp = (x[:, 2] + rad) - zplate
        onp = inb & (penp > 0.0)
        if onp.any():
            fn = np.maximum(self.kwall * penp[onp] - self.cwall * v[onp, 2], 0.0)
            force[onp, 2] -= fn
            vt = v[onp, :2]
            sp = np.linalg.norm(vt, axis=1)
            ft = np.minimum(self.ctan * sp, _FRICTION * fn)
            scale = np.where(sp > 1e-9, ft / np.maximum(sp, 1e-9), 0.0)
            force[onp, 0] -= scale * vt[:, 0]
            force[onp, 1] -= scale * vt[:, 1]

        # --- adhesion of the top layer to the separating plate ----------------
        idx = self.adh
        anchor = self.reference[idx].copy()
        anchor[:, 2] += disp
        vplate = np.array([0.0, 0.0, self.pull if rising else 0.0])
        force[idx] += self.kadh * (anchor - x[idx])
        force[idx] += self.cadh * (vplate[None, :] - v[idx])

    # -------------------------------------------------------------- trajectory
    def _deformation_gradient(self) -> np.ndarray:
        """Moving-least-squares F from the current bond configuration."""
        act = np.flatnonzero(self.active)
        F = np.zeros((self.n, 3, 3), dtype=np.float64)
        if act.size:
            pi = self.pi[act]
            pj = self.pj[act]
            r = self.x[pi] - self.x[pj]
            R = self.reference[pi] - self.reference[pj]
            A = np.zeros((self.n, 3, 3), dtype=np.float64)
            M = np.zeros((self.n, 3, 3), dtype=np.float64)
            for a in range(3):
                for b in range(3):
                    wA = r[:, a] * R[:, b]
                    wM = R[:, a] * R[:, b]
                    A[:, a, b] = np.bincount(pi, weights=wA, minlength=self.n)
                    M[:, a, b] = np.bincount(pi, weights=wM, minlength=self.n)
            tr = np.trace(M, axis1=1, axis2=2)
            eps = 1e-3 * tr + 1e-12
            M = M + eps[:, None, None] * np.eye(3)[None, :, :]
            good = tr > 1e-9
            if good.any():
                sol = np.linalg.solve(M[good], np.transpose(A[good], (0, 2, 1)))
                F[good] = np.transpose(sol, (0, 2, 1))
        F[~np.isfinite(F)] = 0.0
        np.clip(F, -4.0, 4.0, out=F)
        bad = np.linalg.det(F) < 0.05
        F[bad] = np.eye(3)[None, :, :]
        return F

    def _contact_gap(self) -> np.ndarray:
        x = self.x
        floor_gap = x[:, 2] - _FLOOR_TOP
        inb = (np.abs(x[:, 0]) <= _PLATE_HX) & (np.abs(x[:, 1]) <= _PLATE_HY)
        zplate = self._plate_disp(self._time) + _PLATE_BOTTOM0
        plate_gap = np.where(inb, zplate - x[:, 2], np.inf)
        gap = np.minimum(floor_gap, plate_gap)
        keep = max(4, int(np.count_nonzero(gap < 0.08)))
        order = np.argsort(gap, kind="stable")[:keep]
        return np.asarray([min(floor_gap[i], plate_gap[i]) for i in order], dtype=np.float64)

    def _frame(self, time: float) -> StateFrame:
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "deformation_gradient": self._deformation_gradient(),
                "mass_quantity": self.m.copy(),
                "particle_radius": self.rad.copy(),
                "foam_material": np.ones(self.n, dtype=np.float64),
                "tear_candidate": self.tear_candidate.copy(),
                "bond_damage": self._particle_damage(),
            },
            fields={
                "contact_signed_distance": self._contact_gap(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": self.m[:, None] * self.gvec[None, :],
            },
            scalars={"contact_length_scale": float(self.spacing)},
        )
        frame.validate()
        return frame

    def _particle_damage(self) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        cnt = np.zeros(self.n, dtype=np.float64)
        d = self.D[self.active]
        if d.size:
            pi = self.pi[self.active]
            out += np.bincount(pi, weights=d, minlength=self.n)
            cnt += np.bincount(pi, weights=np.ones_like(d), minlength=self.n)
        return out / np.maximum(cnt, 1.0)
