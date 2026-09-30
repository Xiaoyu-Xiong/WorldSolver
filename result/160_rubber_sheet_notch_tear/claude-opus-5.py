"""Notched rubber sheet tear solver (self-implemented, NumPy only).

Physical model
--------------
The sheet is a thin hyperelastic membrane discretised with constant-strain
triangles on the frozen 17x17 grid topology (``assets/cloth_topology.json``),
which already carries the edge notch as a set of removed faces.

* Membrane elasticity: per-element deformation gradient ``F = Ds Dm^{-1}``
  (3x2), Green strain ``E = 1/2 (F^T F - I)`` and a St. Venant-Kirchhoff
  plane-stress second Piola stress ``S = 2 mu E + lambda tr(E) I``.
  Nodal forces come from ``P = F S`` assembled with the element area and
  thickness. Strain-rate (Kelvin-Voigt) damping is added through ``Edot``.
* Bending / flexural resistance: discrete curvature springs on collinear
  vertex triples, gated by edge connectivity so torn material stops
  transmitting bending.
* Fracture: an isotropic damage variable per element driven by the maximum
  principal Green strain. Damage may only grow in elements touching the
  current crack front (initially the notch boundary), which yields a
  connected, self-propagating crack path; a much higher bulk threshold
  allows rupture away from the front. Failed elements lose all stiffness,
  so the two flaps separate while intact material stays coherent. No mass
  is created or destroyed.
* Boundary conditions: the two grip columns (x = -+1.5) are kinematic and
  separate along x with a ramped, capped velocity (prescribed tensile
  loading). Free vertices collide with the ground slab and the grip boxes
  through analytic box signed distances with projection + friction.
* Time integration: fixed-step symplectic (semi-implicit) Euler with a
  substep chosen below the membrane CFL limit. Everything is deterministic.
"""

from __future__ import annotations

import json
import os

import numpy as np

from worldsolver.contracts import StateFrame

_EYE2 = np.eye(2)


def _box_signed_distance(points: np.ndarray, center: np.ndarray, half: np.ndarray):
    """Analytic signed distance and outward gradient for an axis-aligned box."""
    q = np.abs(points - center) - half
    sign = np.sign(points - center)
    sign[sign == 0.0] = 1.0
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    sd = outside + inside
    grad = np.zeros_like(points)
    out_mask = outside > 0.0
    if np.any(out_mask):
        qc = np.maximum(q[out_mask], 0.0)
        norm = np.maximum(np.linalg.norm(qc, axis=1, keepdims=True), 1e-12)
        grad[out_mask] = sign[out_mask] * qc / norm
    in_mask = ~out_mask
    if np.any(in_mask):
        axis = np.argmax(q[in_mask], axis=1)
        g = np.zeros((int(np.count_nonzero(in_mask)), 3))
        g[np.arange(len(axis)), axis] = 1.0
        grad[in_mask] = sign[in_mask] * g
    return sd, grad


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        cloth = cfg["cloth"]

        self.nx, self.ny = int(cloth["grid_shape"][0]), int(cloth["grid_shape"][1])
        self.sx, self.sy = float(cloth["size"][0]), float(cloth["size"][1])
        self.center = np.asarray(cloth["center"], dtype=float)
        self.thickness = float(cloth["thickness"])
        self.total_mass = float(cloth["total_mass"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.duration = float(cfg["duration"])
        self.plane = str(cloth.get("plane", "xy"))

        n = self.nx * self.ny
        self.n = n

        # ---- material (2D) coordinates and initial world positions
        xs = np.linspace(-self.sx * 0.5, self.sx * 0.5, self.nx)
        ys = np.linspace(-self.sy * 0.5, self.sy * 0.5, self.ny)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        self.material = np.column_stack((xx.ravel(), yy.ravel()))
        if self.plane == "xz":
            pos = np.column_stack((xx.ravel(), np.zeros(n), yy.ravel())) + self.center
        else:
            pos = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(n))) + self.center
        self.x0 = pos.copy()
        self.x = pos.copy()
        self.v = np.zeros_like(pos)

        # ---- topology (notched) from the frozen asset
        self.tri = self._load_faces()
        m = len(self.tri)

        # ---- lumped mass (uniform, matches the scaffold initial frame)
        self.mass = np.full(n, self.total_mass / n)
        self.inv_mass = 1.0 / self.mass

        # ---- element rest configuration
        p0 = self.material[self.tri[:, 0]]
        p1 = self.material[self.tri[:, 1]]
        p2 = self.material[self.tri[:, 2]]
        Dm = np.stack((p1 - p0, p2 - p0), axis=2)  # (M,2,2)
        det = Dm[:, 0, 0] * Dm[:, 1, 1] - Dm[:, 0, 1] * Dm[:, 1, 0]
        # keep positive orientation for a well defined material frame
        flip = det < 0.0
        if np.any(flip):
            self.tri[flip] = self.tri[flip][:, [0, 2, 1]]
            p1 = self.material[self.tri[:, 1]]
            p2 = self.material[self.tri[:, 2]]
            Dm = np.stack((p1 - p0, p2 - p0), axis=2)
            det = Dm[:, 0, 0] * Dm[:, 1, 1] - Dm[:, 0, 1] * Dm[:, 1, 0]
        self.rest_area = 0.5 * np.abs(det)
        self.Dm_inv = np.linalg.inv(Dm)

        # ---- rubber-like membrane material
        youngs = 1.2e5
        poisson = 0.3
        self.mu = youngs / (2.0 * (1.0 + poisson))
        self.lam = youngs * poisson / (1.0 - poisson * poisson)  # plane stress
        self.beta = 6.0  # Pa*s strain-rate damping
        self.air_drag = 1.0  # 1/s

        # ---- fracture parameters (Green strain thresholds)
        self.eps_crack = 0.13
        self.eps_bulk = 0.70
        self.damage_rate = 9.0
        self.damage = np.zeros(m)

        # ---- crack front seeded from the notch boundary
        self.crack_vertex = self._seed_crack_front()

        # ---- edge table + bending triples
        self._build_edges()
        self._build_bending()
        self.bend_k = 0.2

        # ---- grips: pinned columns of the sheet
        col = np.arange(n) % self.nx
        self.left_idx = np.where(col == 0)[0]
        self.right_idx = np.where(col == self.nx - 1)[0]
        self.pinned = np.zeros(n, dtype=bool)
        self.pinned[self.left_idx] = True
        self.pinned[self.right_idx] = True
        # Vertices with no surviving incident face (interior of the cut notch)
        # carry mass but no elastic connectivity; hold them inert at rest so the
        # exported state stays coherent and mass is conserved.
        attached = np.zeros(n, dtype=bool)
        attached[self.tri.ravel()] = True
        self.detached = ~attached
        self.free = attached & ~self.pinned
        self.grip_hold = 0.3
        self.grip_ramp = 0.25
        self.grip_speed = 0.10
        self.grip_max = 0.45

        # ---- obstacles (frozen geometry)
        self.obstacles = [
            (np.array([0.0, 0.0, -0.05]), np.array([2.4, 2.0, 0.05]), True),
            (np.array([-1.5, 0.0, 1.6]), np.array([0.12, 0.18, 1.25]), False),
            (np.array([1.5, 0.0, 1.6]), np.array([0.12, 0.18, 1.25]), False),
        ]

        # ---- time stepping
        self.dt = 1.2e-4
        self.time = 0.0
        self.steps = 0
        self._max_speed = 60.0
        self._blowups = 0

    # ---------------------------------------------------------------- helpers
    def _load_faces(self) -> np.ndarray:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(here), "assets", "cloth_topology.json")
        faces = None
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict) and "faces" in data:
                    faces = np.asarray(data["faces"], dtype=np.int64)
            except Exception:
                faces = None
        if faces is None or faces.ndim != 2 or faces.shape[1] != 3:
            faces = self._full_grid_faces()
        return np.ascontiguousarray(faces.astype(np.int64))

    def _full_grid_faces(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        out = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                out.append((a, a + 1, a + nx + 1))
                out.append((a, a + nx + 1, a + nx))
        return np.asarray(out, dtype=np.int64)

    def _seed_crack_front(self) -> np.ndarray:
        """Vertices on the boundary of the pre-existing notch."""
        present = set()
        for t in self.tri:
            present.add(tuple(sorted(int(k) for k in t)))
        notch_v = np.zeros(self.n, dtype=bool)
        for t in self._full_grid_faces():
            if tuple(sorted(int(k) for k in t)) not in present:
                notch_v[t] = True
        attached = np.zeros(self.n, dtype=bool)
        attached[self.tri.ravel()] = True
        return notch_v & attached

    def _build_edges(self) -> None:
        pairs = np.concatenate(
            (self.tri[:, [0, 1]], self.tri[:, [1, 2]], self.tri[:, [2, 0]]), axis=0
        )
        pairs = np.sort(pairs, axis=1)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        self.edges = uniq
        self.n_edges = len(uniq)
        # tri_edge_ids[k] holds edge ids of edge slot k for every triangle
        self.tri_edge_ids = inv.reshape(3, len(self.tri)).T.copy()
        self._edge_lookup = {
            (int(a), int(b)): int(k) for k, (a, b) in enumerate(uniq)
        }

    def _build_bending(self) -> None:
        nx, ny = self.nx, self.ny
        a_list, b_list, c_list, e1, e2 = [], [], [], [], []

        def edge_id(u, v):
            key = (u, v) if u < v else (v, u)
            return self._edge_lookup.get(key, -1)

        for j in range(ny):
            for i in range(nx - 2):
                a = j * nx + i
                b = a + 1
                c = a + 2
                k1, k2 = edge_id(a, b), edge_id(b, c)
                if k1 >= 0 and k2 >= 0:
                    a_list.append(a)
                    b_list.append(b)
                    c_list.append(c)
                    e1.append(k1)
                    e2.append(k2)
        for j in range(ny - 2):
            for i in range(nx):
                a = j * nx + i
                b = a + nx
                c = a + 2 * nx
                k1, k2 = edge_id(a, b), edge_id(b, c)
                if k1 >= 0 and k2 >= 0:
                    a_list.append(a)
                    b_list.append(b)
                    c_list.append(c)
                    e1.append(k1)
                    e2.append(k2)
        self.bend_a = np.asarray(a_list, dtype=np.int64)
        self.bend_b = np.asarray(b_list, dtype=np.int64)
        self.bend_c = np.asarray(c_list, dtype=np.int64)
        self.bend_e1 = np.asarray(e1, dtype=np.int64)
        self.bend_e2 = np.asarray(e2, dtype=np.int64)

    # -------------------------------------------------------------- kinematics
    def _grip_offset(self, t: float) -> tuple[float, float]:
        """Prescribed grip displacement and speed (one side, magnitude)."""
        tau = t - self.grip_hold
        if tau <= 0.0:
            return 0.0, 0.0
        r = self.grip_ramp
        if tau < r:
            disp = 0.5 * self.grip_speed * tau * tau / r
            speed = self.grip_speed * tau / r
        else:
            disp = 0.5 * self.grip_speed * r + self.grip_speed * (tau - r)
            speed = self.grip_speed
        if disp >= self.grip_max:
            disp = self.grip_max
            speed = 0.0
        return disp, speed

    # ------------------------------------------------------------------ forces
    def _element_state(self):
        x = self.x
        tri = self.tri
        d1 = x[tri[:, 1]] - x[tri[:, 0]]
        d2 = x[tri[:, 2]] - x[tri[:, 0]]
        Ds = np.stack((d1, d2), axis=2)  # (M,3,2)
        F = np.einsum("mij,mjk->mik", Ds, self.Dm_inv)  # (M,3,2)
        C = np.einsum("mji,mjk->mik", F, F)  # F^T F
        E = 0.5 * (C - _EYE2)
        return Ds, F, E

    def _forces(self):
        n = self.n
        force = np.zeros((n, 3))
        force += self.mass[:, None] * self.gravity  # external (gravity) load

        Ds, F, E = self._element_state()
        scale = np.clip(1.0 - self.damage, 0.0, 1.0)

        trE = E[:, 0, 0] + E[:, 1, 1]
        S = 2.0 * self.mu * E + self.lam * trE[:, None, None] * _EYE2

        # Kelvin-Voigt strain-rate damping
        vv = self.v
        tri = self.tri
        w1 = vv[tri[:, 1]] - vv[tri[:, 0]]
        w2 = vv[tri[:, 2]] - vv[tri[:, 0]]
        Dsd = np.stack((w1, w2), axis=2)
        Fd = np.einsum("mij,mjk->mik", Dsd, self.Dm_inv)
        A = np.einsum("mji,mjk->mik", Fd, F)
        Edot = 0.5 * (A + np.transpose(A, (0, 2, 1)))
        S = S + self.beta * Edot

        S *= scale[:, None, None]
        P = np.einsum("mij,mjk->mik", F, S)  # (M,3,2)
        H = -(self.rest_area * self.thickness)[:, None, None] * np.einsum(
            "mik,mjk->mij", P, self.Dm_inv
        )
        f1 = H[:, :, 0]
        f2 = H[:, :, 1]
        f0 = -(f1 + f2)
        np.add.at(force, tri[:, 0], f0)
        np.add.at(force, tri[:, 1], f1)
        np.add.at(force, tri[:, 2], f2)

        # bending / flexural springs, gated by surviving connectivity
        if len(self.bend_a):
            alive = (scale > 1e-3).astype(float)
            edge_alive = np.bincount(
                self.tri_edge_ids.ravel(),
                weights=np.repeat(alive, 3),
                minlength=self.n_edges,
            )
            gate = ((edge_alive[self.bend_e1] > 0.5) & (edge_alive[self.bend_e2] > 0.5))
            if np.any(gate):
                ia = self.bend_a[gate]
                ib = self.bend_b[gate]
                ic = self.bend_c[gate]
                lap = self.x[ia] - 2.0 * self.x[ib] + self.x[ic]
                lapv = self.v[ia] - 2.0 * self.v[ib] + self.v[ic]
                q = self.bend_k * lap + 0.004 * lapv
                np.add.at(force, ia, -q)
                np.add.at(force, ib, 2.0 * q)
                np.add.at(force, ic, -q)

        return force, E

    # ------------------------------------------------------------- fracture
    def _update_damage(self, E: np.ndarray, dt: float) -> None:
        half = 0.5 * (E[:, 0, 0] + E[:, 1, 1])
        diff = 0.5 * (E[:, 0, 0] - E[:, 1, 1])
        rad = np.sqrt(diff * diff + E[:, 0, 1] * E[:, 0, 1])
        emax = half + rad

        active = self.damage > 0.0
        eligible = self.crack_vertex[self.tri].any(axis=1) | active
        drive = np.zeros_like(emax)
        exc_tip = emax / self.eps_crack - 1.0
        exc_bulk = emax / self.eps_bulk - 1.0
        drive = np.where(eligible, np.maximum(exc_tip, 0.0), 0.0)
        drive = np.maximum(drive, np.maximum(exc_bulk, 0.0))
        alive = self.damage < 1.0
        grow = np.where(alive, self.damage_rate * np.minimum(drive, 4.0) * dt, 0.0)
        if np.any(grow > 0.0):
            before = self.damage >= 1.0
            self.damage = np.clip(self.damage + grow, 0.0, 1.0)
            newly = (self.damage >= 1.0) & ~before
            if np.any(newly):
                self.crack_vertex[self.tri[newly].ravel()] = True

    # -------------------------------------------------------------- contacts
    def _signed_distances(self):
        half_t = 0.5 * self.thickness
        best = np.full(self.n, np.inf)
        for center, half, applies_to_pinned in self.obstacles:
            sd, _ = _box_signed_distance(self.x, center, half)
            sd = sd - half_t
            if not applies_to_pinned:
                sd = np.where(self.pinned, np.inf, sd)
            best = np.minimum(best, sd)
        finite = np.isfinite(best)
        if not np.all(finite):
            best = np.where(finite, best, 10.0)
        return best

    def _resolve_contacts(self) -> None:
        half_t = 0.5 * self.thickness
        for center, half, applies_to_pinned in self.obstacles:
            sd, grad = _box_signed_distance(self.x, center, half)
            gap = sd - half_t
            mask = gap < 0.0
            if not applies_to_pinned:
                mask &= self.free
            else:
                mask &= self.free
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            nrm = grad[idx]
            norm = np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            nrm = nrm / norm
            self.x[idx] -= gap[idx, None] * nrm
            vn = np.sum(self.v[idx] * nrm, axis=1)
            neg = vn < 0.0
            if np.any(neg):
                sub = idx[neg]
                nn = nrm[neg]
                vt = self.v[sub] - vn[neg, None] * nn
                self.v[sub] = 0.55 * vt  # normal restitution 0, Coulomb-like drag

    # ------------------------------------------------------------------ steps
    def _step(self, dt: float) -> None:
        force, E = self._forces()
        self._update_damage(E, dt)

        acc = force * self.inv_mass[:, None]
        self.v[self.free] += dt * acc[self.free]
        self.v[self.free] /= 1.0 + self.air_drag * dt

        self.time += dt
        disp, speed = self._grip_offset(self.time)

        self.x[self.free] += dt * self.v[self.free]

        # kinematic grips: prescribed tensile separation along x
        self.x[self.left_idx, 0] = self.x0[self.left_idx, 0] - disp
        self.x[self.left_idx, 1] = self.x0[self.left_idx, 1]
        self.x[self.left_idx, 2] = self.x0[self.left_idx, 2]
        self.x[self.right_idx, 0] = self.x0[self.right_idx, 0] + disp
        self.x[self.right_idx, 1] = self.x0[self.right_idx, 1]
        self.x[self.right_idx, 2] = self.x0[self.right_idx, 2]
        self.v[self.left_idx] = np.array([-speed, 0.0, 0.0])
        self.v[self.right_idx] = np.array([speed, 0.0, 0.0])

        self._resolve_contacts()

        # numerical safeguards keep the exported trajectory finite
        if not np.all(np.isfinite(self.x)):
            self.x = np.where(np.isfinite(self.x), self.x, self.x0)
            self.v = np.where(np.isfinite(self.v), self.v, 0.0)
            self._blowups += 1
        if not np.all(np.isfinite(self.v)):
            self.v = np.where(np.isfinite(self.v), self.v, 0.0)
            self._blowups += 1
        speed_mag = np.linalg.norm(self.v, axis=1)
        hot = speed_mag > self._max_speed
        if np.any(hot):
            self.v[hot] *= (self._max_speed / speed_mag[hot])[:, None]

    # ------------------------------------------------------------------ output
    def _vertex_deformation_gradient(self) -> np.ndarray:
        _, F, _ = self._element_state()
        # local orthonormal frame of the deformed element -> 2x2 in-plane F
        c0 = F[:, :, 0]
        c1 = F[:, :, 1]
        n0 = np.maximum(np.linalg.norm(c0, axis=1), 1e-12)
        q1 = c0 / n0[:, None]
        proj = np.sum(c1 * q1, axis=1)
        r = c1 - proj[:, None] * q1
        nr = np.maximum(np.linalg.norm(r, axis=1), 1e-12)
        q2 = r / nr[:, None]
        f2 = np.zeros((len(F), 2, 2))
        f2[:, 0, 0] = n0
        f2[:, 0, 1] = proj
        f2[:, 1, 0] = np.sum(q2 * c0, axis=1)
        f2[:, 1, 1] = nr

        alive = (self.damage < 1.0).astype(float)
        out = np.zeros((self.n, 2, 2))
        weight = np.bincount(
            self.tri.ravel(), weights=np.repeat(alive, 3), minlength=self.n
        )
        for a in range(2):
            for b in range(2):
                acc = np.bincount(
                    self.tri.ravel(),
                    weights=np.repeat(f2[:, a, b] * alive, 3),
                    minlength=self.n,
                )
                out[:, a, b] = acc
        good = weight > 1e-9
        out[good] /= weight[good, None, None]
        out[~good] = _EYE2
        out = np.where(np.isfinite(out), out, 0.0)
        return np.clip(out, -1e4, 1e4)

    def _frame(self) -> StateFrame:
        sd = self._signed_distances()
        ext = self.mass[:, None] * self.gravity
        boundary = self.pinned.astype(np.int32)
        frame = StateFrame(
            time=max(float(self.time), 0.0),
            position=np.ascontiguousarray(self.x.copy()),
            velocity=np.ascontiguousarray(self.v.copy()),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": self._vertex_deformation_gradient(),
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(sd),
                "cumulative_mass_source": np.zeros(1),
                "external_force": np.ascontiguousarray(ext),
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "damaged_element_fraction": float(np.mean(self.damage >= 1.0)),
                "grip_separation": float(
                    self.x[self.right_idx, 0].mean() - self.x[self.left_idx, 0].mean()
                ),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < 0.0:
            target = 0.0
        target_steps = int(round(target / self.dt))
        while self.steps < target_steps:
            self._step(self.dt)
            self.steps += 1
        self.time = self.steps * self.dt
        frame = self._frame()
        frame.time = target
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit StVK membrane FEM with damage-based tearing",
            "dt": float(self.dt),
            "steps": int(self.steps),
            "elements": int(len(self.tri)),
            "vertices": int(self.n),
            "failed_elements": int(np.count_nonzero(self.damage >= 1.0)),
            "nonfinite_repairs": int(self._blowups),
            "total_mass": float(np.sum(self.mass)),
        }
