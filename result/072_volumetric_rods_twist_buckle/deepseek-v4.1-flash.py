"""Physical solver: twist-induced buckling of a seven-rod volumetric bundle.

Physical model
--------------
Each rod is discretised with the frozen surface mesh (49 cross-section rings of
10 surface vertices plus 2 cap centres, 492 nodes per rod).  That surface mesh
is turned into a genuine volumetric-elastic lattice: axial, hoop,
diagonal-shear and cap springs whose stiffnesses follow the standard
``k = E * A_e / L_e`` rule evaluated from the frozen rest geometry.  The axial
springs carry stretch and, through the outer fibres, bending; the hoop/shear
springs carry the shear and torsional rigidity of the rod cross-section.  The
lattice is therefore a real three-dimensional elastic solid discretisation,
not a kinematic animation.

Dynamics are advanced with a semi-implicit (symplectic) Euler time step under
mass-proportional Rayleigh damping.  The two driven end regions are enforced as
moving Dirichlet constraints: they undergo a rigid rotation about the global x
axis at the frozen angular speeds, so the separate rods wind around one another
and, once the twisting exceeds the supercoiling threshold, the bundle writhes
into repeated buckled coils.  Rod-rod interaction is a soft unilateral
repulsion between ring centre-lines (surfaces touch at centre distance
``2 * rod_radius``), which lets neighbouring rods be pressed closely together
without interpenetrating.  Every substep is deterministic and finite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_G_DOMAINS = 8  # seven individual rod bodies plus one combined bundle domain
_K_TERMS = 2  # 0: driven-end constraint reaction, 1: inter-rod contact force


def _load_config(scene) -> dict:
    cfg = getattr(scene, "config", None)
    if isinstance(cfg, dict) and cfg:
        return cfg
    candidates = (
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parents[1] / "config.yaml",
    )
    for candidate in candidates:
        try:
            import yaml

            data = yaml.safe_load(candidate.read_text())
            if isinstance(data, dict) and data:
                return data
        except Exception:  # pragma: no cover - fallback path
            continue
    raise RuntimeError("unable to locate the frozen scene configuration")


def _scatter(idx: np.ndarray, vals: np.ndarray, size: int) -> np.ndarray:
    """Accumulate per-edge vectors into a per-node force array."""
    out = np.empty((size, 3), dtype=float)
    for c in range(3):
        out[:, c] = np.bincount(idx, weights=vals[:, c], minlength=size)
    return out


class BenchmarkSolver:
    """Deterministic volumetric-elastic rod-bundle twist/buckling solver."""

    # -- frozen material / numerical parameters -----------------------------
    YOUNGS_MODULUS = 2.5e5  # Pa, elastic modulus of the rod material
    SHEAR_THICKNESS_RATIO = 0.35  # effective hoop/shear ligament thickness
    DAMPING_RATE = 30.0  # 1/s, mass-proportional Rayleigh damping
    CONTACT_STIFFNESS = 3.0e4  # N/m, soft unilateral surface repulsion
    CONTACT_GAP = 0.0  # m, extra surface approach allowed before repulsion
    CONTACT_MIN_RING_SEP = 3  # ignore self-contact between nearby rings
    MAX_SPEED = 30.0  # m/s, numerical safety ceiling
    SUBSTEPS_PER_SAMPLE = 8

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.config = _load_config(scene)

        frame = scene.initial_state()
        self.rest = np.asarray(frame.position, dtype=float).copy()
        self.body = np.asarray(frame.body_id).astype(np.int64).copy()
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.m_inv = 1.0 / self.mass
        self.driven_left = np.asarray(frame.attributes["driven_left"]).astype(bool)
        self.driven_right = np.asarray(frame.attributes["driven_right"]).astype(bool)
        self.n_nodes = int(self.rest.shape[0])

        geom = self.config["geometry"]
        self.bc = self.config["boundary_conditions"]
        self.nx = int(geom["axial_segments"])
        self.nt = int(geom["radial_segments"])
        self.n_rods = int(geom["rod_count"])
        self.radius = float(geom["rod_radius"])
        self.length = float(geom["rod_length"])
        self.npr = self.nx * self.nt + 2
        self.n_rings = self.nx + 1
        if self.n_nodes != self.n_rods * self.npr:
            raise ValueError("state node count does not match the frozen mesh layout")
        self.ring_radius = float(geom["bundle_ring_radius"])
        self.sample_dt = float(self.config["sample_dt"])

        # -- lattice connectivity (identical for every rod) -----------------
        nx, nt = self.nx, self.nt
        seg = np.repeat(np.arange(nx - 1, dtype=np.int64), nt)
        col = np.tile(np.arange(nt, dtype=np.int64), nx - 1)
        ring = np.repeat(np.arange(nx, dtype=np.int64), nt)
        rcol = np.tile(np.arange(nt, dtype=np.int64), nx)
        cap = np.arange(nt, dtype=np.int64)

        a_loc = np.concatenate(
            [
                seg * nt + col,  # axial
                ring * nt + rcol,  # hoop
                seg * nt + col,  # shear +
                seg * nt + col,  # shear -
                np.full(nt, nx * nt, dtype=np.int64),  # left cap spokes
                np.full(nt, nx * nt + 1, dtype=np.int64),  # right cap spokes
            ]
        )
        b_loc = np.concatenate(
            [
                (seg + 1) * nt + col,
                ring * nt + (rcol + 1) % nt,
                (seg + 1) * nt + (col + 1) % nt,
                (seg + 1) * nt + (col - 1) % nt,
                cap,
                (nx - 1) * nt + cap,
            ]
        )
        counts = [seg.size, ring.size, seg.size, seg.size, nt, nt]

        # -- stiffness calibration from the frozen rest geometry ------------
        area = np.pi * self.radius**2
        dx = self.length / nx
        hoop_len = 2.0 * np.pi * self.radius / nt
        shear_len = float(np.hypot(dx, hoop_len))
        eff_area = self.SHEAR_THICKNESS_RATIO * self.radius * dx
        k_axial = self.YOUNGS_MODULUS * area / (dx * nt)
        k_hoop = self.YOUNGS_MODULUS * eff_area / hoop_len
        k_shear = self.YOUNGS_MODULUS * eff_area / shear_len
        k_loc = np.concatenate(
            [
                np.full(counts[0], k_axial),
                np.full(counts[1], k_hoop),
                np.full(counts[2], k_shear),
                np.full(counts[3], k_shear),
                np.full(counts[4], k_axial),
                np.full(counts[5], k_axial),
            ]
        )

        offsets = (np.arange(self.n_rods, dtype=np.int64) * self.npr)[:, None]
        self.sa = (a_loc[None, :] + offsets).ravel()
        self.sb = (b_loc[None, :] + offsets).ravel()
        self.k = np.tile(k_loc, self.n_rods)
        self.L0 = np.linalg.norm(self.rest[self.sb] - self.rest[self.sa], axis=1)

        # -- ring topology used for contact ---------------------------------
        self.ring_idx = (
            np.arange(self.n_rods, dtype=np.int64)[:, None, None] * self.npr
            + np.arange(self.n_rings, dtype=np.int64)[None, :, None] * nt
            + np.arange(nt, dtype=np.int64)[None, None, :]
        )
        self.ring_flat = self.ring_idx.reshape(self.n_rods * self.n_rings, nt)
        self.ring_rod = np.repeat(np.arange(self.n_rods, dtype=np.int64), self.n_rings)
        self.ring_axial = np.tile(np.arange(self.n_rings, dtype=np.int64), self.n_rods)

        # -- time step from the lattice CFL limit and the sampling interval --
        omega_max = float(np.sqrt(self.k.max() / self.mass.min()))
        self.dt = float(min(self.sample_dt / self.SUBSTEPS_PER_SAMPLE, 0.8 / omega_max))
        self.damp = float(np.exp(-self.DAMPING_RATE * self.dt))
        self.vmax = self.MAX_SPEED

        # -- frozen momentum scale and budget weights -----------------------
        speed = max(
            abs(float(self.bc["left_end_angular_speed_deg_s"])),
            abs(float(self.bc["right_end_angular_speed_deg_s"])),
        )
        v_char = max(1.0e-3, np.deg2rad(speed) * (self.ring_radius + self.radius))
        self.reference_momentum = float(self.mass.sum() * v_char)
        self.weights = np.zeros((_G_DOMAINS, self.n_nodes), dtype=float)
        for r in range(self.n_rods):
            self.weights[r] = (self.body == r).astype(float)
        self.weights[self.n_rods] = 1.0

        # -- dynamic state --------------------------------------------------
        self.P = self.rest.copy()
        self.V = np.zeros_like(self.P)
        self.t = 0.0
        self.cum_drive = np.zeros((self.n_rods, 3))
        self.cum_contact = np.zeros((self.n_rods, 3))
        self.acc_drive = np.zeros((self.n_rods, 3))
        self.acc_contact = np.zeros((self.n_rods, 3))
        self.acc_time = 0.0
        self._ready = True

    # -------------------------------------------------------------- dynamics
    def _spring_forces(self, P: np.ndarray) -> np.ndarray:
        d = P[self.sb] - P[self.sa]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1.0e-12
        f = ((self.k * (dist - self.L0) / dist)[:, None]) * d
        return _scatter(self.sa, f, self.n_nodes) + _scatter(self.sb, -f, self.n_nodes)

    def _contact_forces(self, P: np.ndarray) -> np.ndarray:
        cent = P[self.ring_idx].mean(axis=2).reshape(-1, 3)
        pairs = cKDTree(cent).query_pairs(self.CONTACT_RADIUS, output_type="ndarray")
        Fc = np.zeros_like(P)
        if pairs.size == 0:
            return Fc
        i0, i1 = pairs[:, 0], pairs[:, 1]
        same = self.ring_rod[i0] == self.ring_rod[i1]
        near = np.abs(self.ring_axial[i0] - self.ring_axial[i1]) < self.CONTACT_MIN_RING_SEP
        keep = ~(same & near)
        i0, i1 = i0[keep], i1[keep]
        if i0.size == 0:
            return Fc
        dv = cent[i0] - cent[i1]
        dist = np.sqrt(np.einsum("ij,ij->i", dv, dv)) + 1.0e-12
        pen = self.CONTACT_RADIUS - dist
        act = pen > 0.0
        i0, i1, dv, dist, pen = i0[act], i1[act], dv[act], dist[act], pen[act]
        if i0.size == 0:
            return Fc
        fvec = ((self.CONTACT_STIFFNESS * pen / dist)[:, None]) * dv
        w = 1.0 / self.nt
        idx = np.concatenate([self.ring_flat[i0].ravel(), self.ring_flat[i1].ravel()])
        vals = np.concatenate(
            [np.repeat(w * fvec, self.nt, axis=0), np.repeat(-w * fvec, self.nt, axis=0)],
            axis=0,
        )
        return _scatter(idx, vals, self.n_nodes)

    def _drive_angles(self, t: float) -> tuple[float, float]:
        method = getattr(self.scene, "end_rotation_radians", None)
        if callable(method):
            left, right = method(float(t))
            return float(left), float(right)
        start = float(self.bc["start_time"])
        stop = float(self.bc["stop_time"])
        tc = min(max(float(t), start), stop) - start
        return (
            np.deg2rad(float(self.bc["left_end_angular_speed_deg_s"])) * tc,
            np.deg2rad(float(self.bc["right_end_angular_speed_deg_s"])) * tc,
        )

    def _apply_drive(self, t: float) -> None:
        th_l, th_r = self._drive_angles(t)
        om_l = np.deg2rad(float(self.bc["left_end_angular_speed_deg_s"]))
        om_r = np.deg2rad(float(self.bc["right_end_angular_speed_deg_s"]))
        P, V = self.P, self.V
        for mask, theta, omega in (
            (self.driven_left, th_l, om_l),
            (self.driven_right, th_r, om_r),
        ):
            if not mask.any():
                continue
            y0 = self.rest[mask, 1]
            z0 = self.rest[mask, 2]
            c, s = np.cos(theta), np.sin(theta)
            y = c * y0 - s * z0
            z = s * y0 + c * z0
            P[mask, 0] = self.rest[mask, 0]
            P[mask, 1] = y
            P[mask, 2] = z
            V[mask, 0] = 0.0
            V[mask, 1] = -omega * z
            V[mask, 2] = omega * y

    def _substep(self, h: float) -> None:
        P, V = self.P, self.V
        F = self._spring_forces(P)
        Fc = self._contact_forces(P)
        F += Fc
        V += h * (F * self.m_inv[:, None])
        V *= self.damp
        np.clip(V, -self.vmax, self.vmax, out=V)
        v_free = V.copy()
        P += h * V
        self._apply_drive(self.t)
        # constraint reaction impulse supplied by the driven ends
        dv = V - v_free
        imp = dv * self.mass[:, None]
        drive_imp = np.stack(
            [np.bincount(self.body, weights=imp[:, c], minlength=self.n_rods) for c in range(3)],
            axis=1,
        )
        contact_imp = np.stack(
            [
                np.bincount(
                    self.body, weights=Fc[:, c] * h, minlength=self.n_rods
                )
                for c in range(3)
            ],
            axis=1,
        )
        self.cum_drive += drive_imp
        self.acc_drive += drive_imp
        self.cum_contact += contact_imp
        self.acc_contact += contact_imp
        self.acc_time += h

    # ------------------------------------------------------------ trajectory
    def _deformation_gradient(self) -> np.ndarray:
        rings = self.ring_flat.reshape(self.n_rods, self.n_rings, self.nt)
        P_rings = self.P[rings]
        R_rings = self.rest[rings]

        def frames(V):
            cent = V.mean(axis=2)
            e1 = cent[:, 1:, :] - cent[:, :-1, :]
            e2 = V[:, :-1, 0, :] - cent[:, :-1, :]
            e3 = np.cross(e1, e2)
            return np.stack([e1, e2, e3], axis=-1)

        M = frames(P_rings)
        Mr = frames(R_rings)
        F = M @ np.linalg.inv(Mr)
        return F.reshape(-1, 3, 3)

    def _frame(self) -> StateFrame:
        dt_s = max(self.acc_time, 1.0e-12)
        ext = np.zeros((_G_DOMAINS, _K_TERMS, 3))
        drive = self.acc_drive / dt_s
        contact = self.acc_contact / dt_s
        ext[: self.n_rods, 0] = drive
        ext[: self.n_rods, 1] = contact
        ext[self.n_rods, 0] = drive.sum(axis=0)
        ext[self.n_rods, 1] = contact.sum(axis=0)

        impulse = np.zeros((_G_DOMAINS, 3))
        impulse[: self.n_rods] = self.cum_drive + self.cum_contact
        impulse[self.n_rods] = (self.cum_drive + self.cum_contact).sum(axis=0)

        frame = StateFrame(
            float(self.t),
            self.P.copy(),
            self.V.copy(),
            mass=self.mass.copy(),
            body_id=self.body.copy(),
            attributes={"mass_quantity": self.mass.copy()},
            fields={
                "budget_cumulative_external_impulse": impulse,
                "budget_cumulative_inflow_momentum": np.zeros((_G_DOMAINS, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((_G_DOMAINS, 3)),
                "budget_domain_weights": self.weights,
                "budget_external_force": ext,
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": self._deformation_gradient(),
            },
            scalars={"reference_momentum": self.reference_momentum},
        )
        frame.validate()
        self.acc_drive = np.zeros((self.n_rods, 3))
        self.acc_contact = np.zeros((self.n_rods, 3))
        self.acc_time = 0.0
        return frame

    # --------------------------------------------------------------- runtime
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        target = float(time)
        while self.t < target - 1.0e-12:
            h = min(self.dt, target - self.t)
            self._substep(h)
            self.t += h
        self.t = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene_id": str(self.config.get("scene_id", "")),
            "node_count": int(self.n_nodes),
            "spring_count": int(self.sa.size),
            "ring_count": int(self.ring_rod.size),
            "time_step": float(self.dt),
            "youngs_modulus_Pa": float(self.YOUNGS_MODULUS),
            "damping_rate_1_s": float(self.DAMPING_RATE),
            "contact_centerline_threshold_m": float(self.CONTACT_RADIUS),
            "reference_momentum_kg_m_s": float(self.reference_momentum),
        }

    @property
    def CONTACT_RADIUS(self) -> float:
        return 2.0 * self.radius - self.CONTACT_GAP

