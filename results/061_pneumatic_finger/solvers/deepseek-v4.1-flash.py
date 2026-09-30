"""Pressure-driven bending of a chambered soft finger.

Physically-based finite-element solver written from scratch with NumPy only.

Model
-----
The frozen visual asset (``assets/pneumatic_finger.obj``) is a union of eight
axis-aligned boxes: a thick continuous passive underside (the asymmetric,
strain-limiting wall), six chamber lobes separated by narrow compliant hinges,
and a sealed terminal block.  The physical solid is obtained by voxelising that
union on a uniform grid, hollowing each lobe into an enclosed internal chamber
(the pressurised cavity), and splitting every material voxel into six conforming
linear tetrahedra.

Dynamics
--------
* compressible Neo-Hookean hyperelasticity (total Lagrangian),
* lumped nodal mass, semi-implicit (symplectic) Euler time stepping,
* mass- and stiffness-proportional Rayleigh damping so the driven response is
  quasi-static and the held shape is steady,
* the prescribed chamber pressure is applied as a genuine self-equilibrated
  traction on the enclosed cavity walls (outward from the cavity), ramped from
  ``ramp_start_s`` to ``ramp_end_s`` and held,
* the supported base region (x <= 0.03 m) is fully clamped; the resulting
  constraint reaction is reported in the force budget.

The 64 render vertices of the frozen asset are embedded in the tetrahedral mesh
with barycentric coordinates and advected with it, so the exported
StateFrame carries the deformed visual mesh while the physics lives on the
tetrahedral mesh.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# frozen geometry (mirrors assets/generate_pneumatic_finger.py)
# --------------------------------------------------------------------------
_XMAX = 0.220
_YMIN, _YMAX = -0.024, 0.024
_ZMAX = 0.050
_PASSIVE = (0.0, 0.210, -0.020, 0.020, 0.0, 0.014)
_TIP = (0.200, 0.220, -0.020, 0.020, 0.008, 0.038)
_LOBES = []
for _i in range(6):
    _x0 = 0.008 + _i * 0.032
    _LOBES.append((_x0, _x0 + 0.027, -0.024, 0.024, 0.014, 0.050))
_WALL = 0.006  # chamber wall thickness (one voxel)
_CAVITIES = [
    (b[0] + _WALL, b[1] - _WALL, b[2] + _WALL, b[3] - _WALL, b[4] + _WALL, b[5] - _WALL)
    for b in _LOBES
]

# --------------------------------------------------------------------------
# material / discretisation constants
# --------------------------------------------------------------------------
_H = 0.006            # voxel size [m]
_YM = 0.30e6          # Young's modulus [Pa] (soft silicone)
_NU = 0.40            # Poisson ratio
_RHO = 1050.0         # density [kg/m^3]
_MU = _YM / (2.0 * (1.0 + _NU))
_KAPPA = _YM / (3.0 * (1.0 - 2.0 * _NU))
_CWAVE = float(np.sqrt((_KAPPA + 4.0 * _MU / 3.0) / _RHO))
_CFL = 0.25
_ALPHA_DAMP = 30.0    # mass-proportional damping [1/s]
_TAU_DAMP = 0.4       # stiffness-proportional damping time (x dt)
_FIXED_X = 0.0301     # clamped base region (frozen base_support_region_x)
_J_FLOOR = 0.15       # volumetric safeguard on det(F)
_I3 = np.eye(3)

_TET_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
_TET_OPP = np.array([3, 2, 1, 0], dtype=np.int64)
_EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _det3(a: np.ndarray) -> np.ndarray:
    return (
        a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
        - a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
        + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0])
    )


def _inv3(a: np.ndarray) -> np.ndarray:
    d = _det3(a)
    safe = np.where(np.abs(d) < 1e-30, 1e-30, d)
    inv = np.empty_like(a)
    inv[..., 0, 0] = a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1]
    inv[..., 0, 1] = a[..., 0, 2] * a[..., 2, 1] - a[..., 0, 1] * a[..., 2, 2]
    inv[..., 0, 2] = a[..., 0, 1] * a[..., 1, 2] - a[..., 0, 2] * a[..., 1, 1]
    inv[..., 1, 0] = a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2]
    inv[..., 1, 1] = a[..., 0, 0] * a[..., 2, 2] - a[..., 0, 2] * a[..., 2, 0]
    inv[..., 1, 2] = a[..., 0, 2] * a[..., 1, 0] - a[..., 0, 0] * a[..., 1, 2]
    inv[..., 2, 0] = a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0]
    inv[..., 2, 1] = a[..., 0, 1] * a[..., 2, 0] - a[..., 0, 0] * a[..., 2, 1]
    inv[..., 2, 2] = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    return inv / safe[..., None, None]


def _inside(box, x, y, z):
    return (x >= box[0]) & (x <= box[1]) & (y >= box[2]) & (y <= box[3]) & (z >= box[4]) & (z <= box[5])


class BenchmarkSolver:
    """Explicit-dynamics tetrahedral FEM solver for the chambered soft finger."""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", {}) or {}
        pressure = config.get("pressure", {}) or {}
        self._seed = int(seed)
        self._p_target = float(pressure.get("chamber_pressure_pa", 18000.0))
        self._p_start = float(pressure.get("ramp_start_s", 0.40))
        self._p_end = float(pressure.get("ramp_end_s", 2.20))

        initial = scene.initial_state()
        self._rest_visual = np.asarray(initial.position, dtype=float).copy()
        self._n_visual = int(self._rest_visual.shape[0])

        self._build_mesh()
        self._build_pressure_loads()
        self._embed_visual_vertices()

        # mass assigned to the exported visual samples (constant, sums to solid mass)
        wm = self._W @ self._mnode
        total = float(wm.sum())
        if total <= 0.0:
            wm = np.ones(self._n_visual)
            total = float(wm.sum())
        self._visual_mass = self._solid_mass * (wm / total)

        self._x = self._X.copy()
        self._v = np.zeros_like(self._X)
        self._t = 0.0
        self._dt = self._dt_max
        self._steps = 0
        self._rollbacks = 0
        self._min_detF = 1.0
        self._max_speed = 0.0
        self._init_state = {
            "nodes": int(self._X.shape[0]),
            "elements": int(self._T.shape[0]),
            "voxels": int(self._cells.shape[0]),
            "voxel_size_m": _H,
            "solid_mass_kg": float(self._solid_mass),
            "cavity_faces": int(self._cavity_face_count),
            "cavity_area_m2": float(self._cavity_area),
            "stable_dt_s": float(self._dt_max),
            "youngs_modulus_pa": _YM,
            "poisson_ratio": _NU,
            "density_kg_m3": _RHO,
            "reference_force_n": float(self._reference_force),
        }

    # -- mesh ----------------------------------------------------------
    def _build_mesh(self) -> None:
        h = _H
        nx = max(1, int(np.ceil((_XMAX - 0.0) / h)))
        ny = max(1, int(np.ceil((_YMAX - _YMIN) / h)))
        nz = max(1, int(np.ceil((_ZMAX - 0.0) / h)))
        xc = (np.arange(nx) + 0.5) * h
        yc = _YMIN + (np.arange(ny) + 0.5) * h
        zc = (np.arange(nz) + 0.5) * h
        gx, gy, gz = np.meshgrid(xc, yc, zc, indexing="ij")

        solid = np.zeros(gx.shape, dtype=bool)
        for box in [_PASSIVE, *_LOBES, _TIP]:
            solid |= _inside(box, gx, gy, gz)
        cavity = np.zeros(gx.shape, dtype=bool)
        for box in _CAVITIES:
            cavity |= _inside(box, gx, gy, gz)
        material = solid & ~cavity

        cells = np.argwhere(material)
        if cells.shape[0] == 0:
            raise ValueError("voxelisation produced an empty solid")
        self._cells = cells.astype(np.int64)

        index: dict[tuple[int, int, int], int] = {}
        coords: list[tuple[float, float, float]] = []
        cell_nodes = np.empty((cells.shape[0], 8), dtype=np.int64)

        def node_id(i: int, j: int, k: int) -> int:
            key = (i, j, k)
            found = index.get(key)
            if found is None:
                found = len(coords)
                index[key] = found
                coords.append((i * h, _YMIN + j * h, k * h))
            return found

        for c in range(cells.shape[0]):
            i, j, k = int(cells[c, 0]), int(cells[c, 1]), int(cells[c, 2])
            cell_nodes[c] = (
                node_id(i, j, k),
                node_id(i + 1, j, k),
                node_id(i + 1, j + 1, k),
                node_id(i, j + 1, k),
                node_id(i, j, k + 1),
                node_id(i + 1, j, k + 1),
                node_id(i + 1, j + 1, k + 1),
                node_id(i, j + 1, k + 1),
            )

        self._X = np.asarray(coords, dtype=float)
        self._n_nodes = self._X.shape[0]

        # six-tet Freudenthal split (conforming across neighbouring voxels)
        tet_local = (
            (0, 1, 2, 6),
            (0, 2, 3, 6),
            (0, 3, 7, 6),
            (0, 7, 4, 6),
            (0, 4, 5, 6),
            (0, 5, 1, 6),
        )
        tets = np.empty((cell_nodes.shape[0] * 6, 4), dtype=np.int64)
        for t, quad in enumerate(tet_local):
            tets[t::6] = cell_nodes[:, list(quad)]
        self._T = tets

        e1 = self._X[tets[:, 1]] - self._X[tets[:, 0]]
        e2 = self._X[tets[:, 2]] - self._X[tets[:, 0]]
        e3 = self._X[tets[:, 3]] - self._X[tets[:, 0]]
        dm = np.stack((e1, e2, e3), axis=2)
        det = _det3(dm)
        flip = det < 0.0
        if np.any(flip):
            tets[flip, 1], tets[flip, 2] = tets[flip, 2].copy(), tets[flip, 1].copy()
            e1 = self._X[tets[:, 1]] - self._X[tets[:, 0]]
            e2 = self._X[tets[:, 2]] - self._X[tets[:, 0]]
            e3 = self._X[tets[:, 3]] - self._X[tets[:, 0]]
            dm = np.stack((e1, e2, e3), axis=2)
            det = _det3(dm)
        if np.any(det <= 0.0):
            raise ValueError("degenerate tetrahedron in rest mesh")
        self._T = tets
        self._vol = det / 6.0
        inv_dm = _inv3(dm)
        grad = np.empty((tets.shape[0], 4, 3), dtype=float)
        grad[:, 1, :] = inv_dm[:, 0, :]
        grad[:, 2, :] = inv_dm[:, 1, :]
        grad[:, 3, :] = inv_dm[:, 2, :]
        grad[:, 0, :] = -(grad[:, 1, :] + grad[:, 2, :] + grad[:, 3, :])
        self._grad = grad

        element_mass = _RHO * self._vol
        self._solid_mass = float(element_mass.sum())
        mn = np.bincount(self._T.ravel(), weights=np.repeat(element_mass / 4.0, 4),
                         minlength=self._n_nodes)
        self._mnode = mn

        edge_min = np.inf
        for a, b in _EDGE_PAIRS:
            seg = self._X[self._T[:, a]] - self._X[self._T[:, b]]
            edge_min = min(edge_min, float(np.sqrt((seg * seg).sum(axis=1)).min()))
        self._h_min = edge_min
        self._dt_max = _CFL * self._h_min / _CWAVE
        if not np.isfinite(self._dt_max) or self._dt_max <= 0.0:
            self._dt_max = 1e-5

        self._fixed = self._X[:, 0] <= _FIXED_X
        if not np.any(self._fixed):
            self._fixed[:] = self._X[:, 0] <= self._X[:, 0].min() + 1e-9

    # -- pressure load --------------------------------------------------
    def _build_pressure_loads(self) -> None:
        T = self._T
        faces = T[:, _TET_FACES]                       # (Ne,4,3)
        opp = T[:, _TET_OPP]                           # (Ne,4)
        n_faces = faces.shape[0] * 4
        f_nodes = faces.reshape(n_faces, 3)
        f_opp = opp.reshape(n_faces)

        key = np.sort(f_nodes, axis=1)
        n = self._n_nodes
        codes = (key[:, 0] * n + key[:, 1]) * n + key[:, 2]
        _, first, counts = np.unique(codes, return_index=True, return_counts=True)
        boundary = first[counts == 1]
        b_nodes = f_nodes[boundary]
        b_opp = f_opp[boundary]

        p0 = self._X[b_nodes[:, 0]]
        p1 = self._X[b_nodes[:, 1]]
        p2 = self._X[b_nodes[:, 2]]
        d = self._X[b_opp]
        cr = np.cross(p1 - p0, p2 - p0)
        nrm = np.sqrt((cr * cr).sum(axis=1))
        toward = np.sign(((d - p0) * cr).sum(axis=1))
        toward = np.where(toward == 0.0, 1.0, toward)
        # unit normal must point away from the tetrahedron interior (node d)
        cr = cr * (-toward)[:, None]
        nrm = np.maximum(nrm, 1e-30)
        unit = cr / nrm[:, None]
        area = 0.5 * nrm

        centroid = (p0 + p1 + p2) / 3.0 + unit * (0.5 * self._h_min)
        in_cavity = np.zeros(b_nodes.shape[0], dtype=bool)
        for box in _CAVITIES:
            in_cavity |= _inside(box, centroid[:, 0], centroid[:, 1], centroid[:, 2])

        self._cavity_face_count = int(in_cavity.sum())
        sel = b_nodes[in_cavity]
        wgt = -unit[in_cavity] * (area[in_cavity] / 3.0)[:, None]
        self._cavity_area = float(area[in_cavity].sum())

        pcoef = np.zeros((self._n_nodes, 3), dtype=float)
        if sel.shape[0]:
            for c in range(3):
                pcoef[:, c] = np.bincount(
                    sel.ravel(),
                    weights=np.broadcast_to(wgt[:, c][:, None], sel.shape).ravel(),
                    minlength=self._n_nodes,
                )
        self._pcoef = pcoef
        self._pcoef_total = pcoef.sum(axis=0)
        self._reference_force = max(self._p_target * self._cavity_area, 1e-6)

    # -- embed the 64 visual vertices ------------------------------------
    def _embed_visual_vertices(self) -> None:
        T = self._T
        X0 = self._X[T[:, 0]]
        cols = (
            self._X[T[:, 1]] - X0,
            self._X[T[:, 2]] - X0,
            self._X[T[:, 3]] - X0,
        )
        bmat = np.stack(cols, axis=2)
        binv = _inv3(bmat)

        n_vis = self._rest_visual.shape[0]
        w_idx = np.empty((n_vis, 4), dtype=np.int64)
        w_val = np.empty((n_vis, 4), dtype=float)
        for v in range(n_vis):
            rhs = self._rest_visual[v] - X0
            xi = np.einsum("eij,ej->ei", binv, rhs)
            w = np.column_stack((1.0 - xi.sum(axis=1), xi))
            best = int(np.argmax(w.min(axis=1)))
            wk = np.maximum(w[best], 0.0)
            s = float(wk.sum())
            if s <= 1e-12:
                wk = np.array([1.0, 0.0, 0.0, 0.0])
                w_idx[v] = T[best]
            else:
                wk = wk / s
                w_idx[v] = T[best]
            w_val[v] = wk
        self._W_idx = w_idx
        self._W_val = w_val
        wdense = np.zeros((n_vis, self._n_nodes), dtype=float)
        rows = np.repeat(np.arange(n_vis), 4)
        np.add.at(wdense, (rows, w_idx.ravel()), w_val.ravel())
        self._W = wdense

    # ------------------------------------------------------------------
    # pressure schedule
    # ------------------------------------------------------------------
    def _pressure(self, t: float) -> float:
        if t <= self._p_start:
            return 0.0
        if t >= self._p_end:
            return self._p_target
        u = (t - self._p_start) / max(self._p_end - self._p_start, 1e-12)
        u = min(max(u, 0.0), 1.0)
        return self._p_target * (u * u * (3.0 - 2.0 * u))

    # ------------------------------------------------------------------
    # element assembly / time stepping
    # ------------------------------------------------------------------
    def _internal_forces(self, x: np.ndarray, v: np.ndarray, dt: float):
        T = self._T
        grad = self._grad
        xe = x[T]
        ve = v[T]
        F = np.einsum("eai,eaj->eij", xe, grad)
        Fdot = np.einsum("eai,eaj->eij", ve, grad)
        J = _det3(F)
        invF = _inv3(F)
        FinvT = np.swapaxes(invF, 1, 2)
        Js = np.maximum(J, _J_FLOOR)
        P = _MU * (F - FinvT) + (_KAPPA * np.log(Js))[:, None, None] * FinvT

        tau = _TAU_DAMP * dt
        L = Fdot @ invF
        D = 0.5 * (L + np.swapaxes(L, 1, 2))
        trD = D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]
        Ddev = D - (trD / 3.0)[:, None, None] * _I3
        sig = 2.0 * (tau * _MU) * Ddev + (tau * _KAPPA * trD)[:, None, None] * _I3
        Pv = J[:, None, None] * (sig @ FinvT)

        fe = -np.einsum("eij,eaj->eai", P + Pv, grad) * self._vol[:, None, None]
        return fe.reshape(-1, 3), F, J

    def _scatter(self, f_el: np.ndarray) -> np.ndarray:
        n = self._n_nodes
        f = np.empty((n, 3), dtype=float)
        idx = self._T.ravel()
        for c in range(3):
            f[:, c] = np.bincount(idx, weights=f_el[:, c], minlength=n)
        return f

    def _substep(self, dt: float) -> bool:
        x_prev = self._x.copy()
        v_prev = self._v.copy()
        p = self._pressure(self._t)

        f_el, F, J = self._internal_forces(self._x, self._v, dt)
        f_int = self._scatter(f_el)
        f_ext = p * self._pcoef
        f_tot = f_int + f_ext
        a = f_tot / self._mnode[:, None]
        if _ALPHA_DAMP:
            a -= _ALPHA_DAMP * self._v
        a[self._fixed] = 0.0

        v_new = self._v + a * dt
        v_new[self._fixed] = 0.0
        x_new = self._x + v_new * dt
        x_new[self._fixed] = x_prev[self._fixed]

        if not (np.isfinite(v_new).all() and np.isfinite(x_new).all()):
            self._x = x_prev
            self._v = v_prev
            self._dt = max(self._dt * 0.5, 1e-9)
            self._rollbacks += 1
            return False

        self._x = x_new
        self._v = v_new
        self._t += dt
        self._steps += 1
        self._min_detF = min(self._min_detF, float(J.min()))
        speed = float(np.sqrt((self._v * self._v).sum(axis=1)).max())
        self._max_speed = max(self._max_speed, speed)
        return True

    def _advance(self, target: float) -> None:
        guard = 0
        while self._t < target - 1e-13:
            remaining = target - self._t
            n_sub = max(1, int(np.ceil(remaining / self._dt)))
            dt = remaining / n_sub
            failed = False
            for _ in range(n_sub):
                if not self._substep(dt):
                    failed = True
                    break
            if failed:
                guard += 1
                if guard > 200:
                    break
        if self._t > target:
            self._t = target

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target > self._t + 1e-13:
            self._advance(target)
        return self._frame(target)

    def _frame(self, time: float):
        T = self._T
        x = self._x
        v = self._v
        F = np.einsum("eai,eaj->eij", x[T], self._grad)
        J = _det3(F)

        pos = self._W @ x
        vel = self._W @ v

        p = self._pressure(self._t)
        f_el, _F, _J = self._internal_forces(x, v, self._dt)
        f_int = self._scatter(f_el)
        f_ext = p * self._pcoef
        react = -(f_int + f_ext)[self._fixed].sum(axis=0)

        budget = np.zeros((2, 2, 3), dtype=float)
        budget[0, 0] = p * self._pcoef_total          # pressure traction on the finger
        budget[0, 1] = react                          # support reaction on the finger
        budget[1, 1] = -react                         # finger reaction on the support

        attributes = {
            "mass_quantity": self._visual_mass,
            "fixed_base": (self._rest_visual[:, 0] <= 0.030).astype(float),
            "rest_position": self._rest_visual.copy(),
        }
        fields = {
            "element_deformation_gradient": F,
            "budget_external_force": budget,
            "cumulative_mass_source": np.zeros(2, dtype=float),
        }
        scalars = {
            "reference_force": float(self._reference_force),
            "chamber_pressure_pa": float(p),
            "min_det_deformation_gradient": float(J.min()),
        }
        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=self._visual_mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        tip = self._W @ self._x
        tip_delta = float(np.linalg.norm(tip[-1] - self._rest_visual[-1]))
        report = dict(self._init_state)
        report.update(
            {
                "steps": int(self._steps),
                "rollbacks": int(self._rollbacks),
                "final_time_s": float(self._t),
                "final_dt_s": float(self._dt),
                "min_det_deformation_gradient": float(self._min_detF),
                "max_node_speed_m_s": float(self._max_speed),
                "tip_displacement_m": tip_delta,
                "clamped_nodes": int(self._fixed.sum()),
            }
        )
        return report
