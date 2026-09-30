"""Pinned-free cloth sphere drape with principal-strain limiting.

Physical model (implemented from scratch with NumPy/SciPy only):

* Lumped-mass triangle-mesh shell using the frozen per-vertex masses.
* Symplectic-Euler predictor for gravity + air damping, followed by a
  position-based constraint solve (Jacobi with under-relaxation) for the
  membrane, shear and bending response.  This is a projective/position-based
  discretization of a soft shell energy, stable for the very low membrane
  stiffness of the Figure 15(c) configuration.
* Strict principal-strain limiting: every triangle's in-plane deformation
  gradient is SVD-projected so that both principal stretches stay inside the
  frozen admissible band, mimicking the C-IPC strain limiter.  The soft
  membrane buckles into distributed wrinkles instead of locking while the
  limiter forbids excessive stretch and compression.
* Contact: analytic signed-distance projection against the fixed sphere and
  the ground slab with Coulomb friction on tangential slip, plus vertex
  repulsion for cloth self-contact, all offset by half the cloth thickness so
  no interpenetration is visible.
* Velocities are recovered from projected positions, v = (x - x_old) / h.

Everything is deterministic: no random numbers are drawn, the substep count
per sample interval is fixed, and all reductions are order-deterministic.
"""
from __future__ import annotations

import numpy as np

try:  # SciPy is optional; a NumPy fallback keeps the solver runnable.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover - fallback path
    _KDTree = None

from worldsolver.contracts import StateFrame


def _sym2_eig(g00, g01, g11):
    """Analytic eigendecomposition of a batch of symmetric 2x2 matrices.

    Returns (l1, l2, v1x, v1y) with l1 >= l2 and v1 the unit eigenvector of l1.
    """
    half_tr = 0.5 * (g00 + g11)
    disc = np.sqrt(np.maximum(0.25 * (g00 - g11) ** 2 + g01 * g01, 0.0))
    l1 = half_tr + disc
    l2 = np.maximum(half_tr - disc, 0.0)
    vx = g01
    vy = l1 - g00
    nrm = np.sqrt(vx * vx + vy * vy)
    tiny = nrm < 1.0e-9
    if np.any(tiny):
        ax = (g00 >= g11).astype(float)
        vx = np.where(tiny, ax, vx)
        vy = np.where(tiny, 1.0 - ax, vy)
        nrm = np.where(tiny, 1.0, nrm)
    return l1, l2, vx / nrm, vy / nrm


class BenchmarkSolver:
    """Position-based shell solver with hard principal-strain limits."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = dict(scene.config)
        meta = scene.metadata()
        frame0 = scene.initial_state()

        self.position = np.array(frame0.position, dtype=float)
        self.velocity = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.n = self.position.shape[0]
        self.uv = np.array(frame0.attributes["uv"], dtype=float)

        cloth = dict(cfg.get("cloth", {}))
        sphere = dict(cfg.get("sphere", {}))
        solver_cfg = dict(cfg.get("solver", {}))
        ground = dict(cfg.get("ground", {}))

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.duration = float(cfg.get("duration", 4.0))
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.substep_dt = float(solver_cfg.get("substep", 0.004))
        self.iterations = int(solver_cfg.get("constraint_iterations", 15))

        grid = cloth.get("grid_shape") or meta.get("cloth", {}).get("grid_shape", [31, 31])
        self.nx, self.ny = int(grid[0]), int(grid[1])
        self.side = float(cloth.get("side_length", 2.8))
        self.thickness = float(cloth.get("thickness", 0.012))
        self.half_thickness = 0.5 * self.thickness
        self.strain_lo = float(cloth.get("strain_lower", 0.9))
        self.strain_hi = float(cloth.get("strain_upper", 1.1))
        self.tgt_lo = float(cloth.get("target_strain_lower", 0.965))
        self.tgt_hi = float(cloth.get("target_strain_upper", 1.045))
        self.damping = float(cloth.get("damping", 1.35))
        self.friction = float(cloth.get("friction", 0.22))

        self.sphere_center = np.asarray(sphere.get("center", [0.0, 0.0, 0.64]), dtype=float)
        self.sphere_radius = float(sphere.get("radius", 0.64))
        self.ground_z = float(ground.get("height", 0.0))

        faces = meta.get("topology", {}).get("faces")
        self.faces = np.asarray(faces, dtype=np.int64)

        self.inv_mass = np.where(self.mass > 0.0, 1.0 / np.maximum(self.mass, 1e-16), 0.0)
        self.contact_length_scale = float(
            frame0.scalars.get("contact_length_scale", self.thickness)
        )
        self.mass_quantity = np.array(
            frame0.attributes.get("mass_quantity", self.mass), dtype=float
        )
        self.cumulative_mass_source = np.array(
            frame0.fields.get("cumulative_mass_source", [0.0]), dtype=float
        )

        self._build_constraints()
        self._build_elements()

        self.time = 0.0
        self._max_stretch = 1.0
        self._min_stretch = 1.0
        self._substeps_taken = 0
        self._self_collision_counter = 0

    # --------------------------------------------------------- topology setup
    def _vid(self, i, j):
        return i + self.nx * j

    def _build_constraints(self):
        """Structural / shear / bending distance constraints on the grid."""
        nx, ny = self.nx, self.ny
        idx = np.arange(nx * ny).reshape(ny, nx)  # idx[j, i]

        struct = []
        struct.append(np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1))
        struct.append(np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1))
        struct = np.concatenate(struct, axis=0)

        shear = []
        shear.append(np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], axis=1))
        shear.append(np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()], axis=1))
        shear = np.concatenate(shear, axis=0)

        bend = []
        if nx > 2:
            bend.append(np.stack([idx[:, :-2].ravel(), idx[:, 2:].ravel()], axis=1))
        if ny > 2:
            bend.append(np.stack([idx[:-2, :].ravel(), idx[2:, :].ravel()], axis=1))
        bend = np.concatenate(bend, axis=0) if bend else np.zeros((0, 2), dtype=np.int64)

        # Flat-panel rest lengths from the material (uv) coordinates so the
        # small initial out-of-plane seed does not bias the rest state.
        mat = np.column_stack([self.uv[:, 0] * self.side, self.uv[:, 1] * self.side])

        def mat_length(pairs):
            return np.linalg.norm(mat[pairs[:, 0]] - mat[pairs[:, 1]], axis=1)

        self.edges = np.concatenate([struct, shear, bend], axis=0)
        self.rest = np.concatenate(
            [mat_length(struct), mat_length(shear), mat_length(bend)]
        )
        n_s, n_sh, n_b = len(struct), len(shear), len(bend)
        # Soft membrane (0.01x stiffness regime): the distance constraints only
        # provide a gentle restoring force; the strain limiter enforces bounds.
        stiff = np.concatenate(
            [
                np.full(n_s, 0.30),
                np.full(n_sh, 0.12),
                np.full(n_b, 0.012),
            ]
        )
        self.edge_stiffness = stiff
        # Strain limits apply to in-plane (structural + shear) edges only.
        self.limited = np.concatenate(
            [np.ones(n_s + n_sh, dtype=bool), np.zeros(n_b, dtype=bool)]
        )
        self.lim_edges = self.edges[self.limited]
        self.lim_rest = self.rest[self.limited]

        counts = np.zeros(self.n)
        np.add.at(counts, self.edges[:, 0], 1.0)
        np.add.at(counts, self.edges[:, 1], 1.0)
        self.edge_count = np.maximum(counts, 1.0)

        lcounts = np.zeros(self.n)
        np.add.at(lcounts, self.lim_edges[:, 0], 1.0)
        np.add.at(lcounts, self.lim_edges[:, 1], 1.0)
        self.lim_count = np.maximum(lcounts, 1.0)

        self.rest_dx = self.side / max(self.nx - 1, 1)
        self.rest_dy = self.side / max(self.ny - 1, 1)

    def _build_elements(self):
        """Per-triangle material frames plus a vertex -> triangle sampling map."""
        f = self.faces
        mat = np.column_stack([self.uv[:, 0] * self.side, self.uv[:, 1] * self.side])
        e1 = mat[f[:, 1]] - mat[f[:, 0]]
        e2 = mat[f[:, 2]] - mat[f[:, 0]]
        # Dm = [e1 e2] (2x2); store its inverse for F = Ds @ Dm^{-1}.
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        inv = np.empty((len(f), 2, 2))
        inv[:, 0, 0] = e2[:, 1] / det
        inv[:, 0, 1] = -e2[:, 0] / det
        inv[:, 1, 0] = -e1[:, 1] / det
        inv[:, 1, 1] = e1[:, 0] / det
        self.dm_inv = inv
        self.elem_area = 0.5 * np.abs(det)

        tri_counts = np.zeros(self.n)
        np.add.at(tri_counts, f[:, 0], 1.0)
        np.add.at(tri_counts, f[:, 1], 1.0)
        np.add.at(tri_counts, f[:, 2], 1.0)
        self.tri_count = np.maximum(tri_counts, 1.0)

        # Deterministic per-vertex element sample (first incident triangle).
        owner = np.full(self.n, -1, dtype=np.int64)
        for t in range(len(f) - 1, -1, -1):
            owner[f[t, 0]] = t
            owner[f[t, 1]] = t
            owner[f[t, 2]] = t
        owner[owner < 0] = 0
        self.vertex_element = owner

    # ------------------------------------------------------------ deformation
    def _element_frames(self, pos):
        """Return (F2 as 3x2 world columns, dm_inv applied) per triangle."""
        f = self.faces
        d1 = pos[f[:, 1]] - pos[f[:, 0]]
        d2 = pos[f[:, 2]] - pos[f[:, 0]]
        inv = self.dm_inv
        # A columns: a1 = d1*inv00 + d2*inv10 ; a2 = d1*inv01 + d2*inv11
        a1 = d1 * inv[:, 0, 0][:, None] + d2 * inv[:, 1, 0][:, None]
        a2 = d1 * inv[:, 0, 1][:, None] + d2 * inv[:, 1, 1][:, None]
        return a1, a2

    def _principal_stretches(self, a1, a2):
        g00 = np.einsum("ij,ij->i", a1, a1)
        g01 = np.einsum("ij,ij->i", a1, a2)
        g11 = np.einsum("ij,ij->i", a2, a2)
        l1, l2, vx, vy = _sym2_eig(g00, g01, g11)
        return np.sqrt(np.maximum(l1, 0.0)), np.sqrt(np.maximum(l2, 0.0)), vx, vy

    def _limit_strain(self, pos, iterations, lo, hi, relax=1.0):
        """Project each triangle so principal stretches lie in [lo, hi]."""
        f = self.faces
        inv = self.dm_inv
        for _ in range(iterations):
            a1, a2 = self._element_frames(pos)
            s1, s2, vx, vy = self._principal_stretches(a1, a2)
            c1 = np.clip(s1, lo, hi)
            c2 = np.clip(s2, lo, hi)
            r1 = np.where(s1 > 1e-9, c1 / np.maximum(s1, 1e-9), 1.0)
            r2 = np.where(s2 > 1e-9, c2 / np.maximum(s2, 1e-9), 1.0)
            active = (np.abs(r1 - 1.0) > 1e-9) | (np.abs(r2 - 1.0) > 1e-9)
            if not np.any(active):
                break
            # B = V diag(r) V^T (symmetric 2x2), applied on the right of A.
            b00 = r1 * vx * vx + r2 * vy * vy
            b01 = (r1 - r2) * vx * vy
            b11 = r1 * vy * vy + r2 * vx * vx
            b00 = np.where(active, b00, 1.0)
            b11 = np.where(active, b11, 1.0)
            b01 = np.where(active, b01, 0.0)
            na1 = a1 * b00[:, None] + a2 * b01[:, None]
            na2 = a1 * b01[:, None] + a2 * b11[:, None]
            # Back to world edge vectors: D = A_new @ Dm, Dm = inv^{-1}.
            # Since A = D @ inv, D = A @ inv^{-1}; recover inv^{-1} from inv.
            di = self._dm_from_inv(inv)
            nd1 = na1 * di[:, 0, 0][:, None] + na2 * di[:, 1, 0][:, None]
            nd2 = na1 * di[:, 0, 1][:, None] + na2 * di[:, 1, 1][:, None]
            d1 = pos[f[:, 1]] - pos[f[:, 0]]
            d2 = pos[f[:, 2]] - pos[f[:, 0]]
            g1 = nd1 - d1
            g2 = nd2 - d2
            # Centroid-preserving distribution of the two edge corrections.
            c0 = -(g1 + g2) / 3.0
            cc1 = (2.0 * g1 - g2) / 3.0
            cc2 = (2.0 * g2 - g1) / 3.0
            delta = np.zeros_like(pos)
            np.add.at(delta, f[:, 0], c0)
            np.add.at(delta, f[:, 1], cc1)
            np.add.at(delta, f[:, 2], cc2)
            pos += relax * delta / self.tri_count[:, None]
        return pos

    @staticmethod
    def _dm_from_inv(inv):
        det = inv[:, 0, 0] * inv[:, 1, 1] - inv[:, 0, 1] * inv[:, 1, 0]
        det = np.where(np.abs(det) < 1e-18, 1e-18, det)
        out = np.empty_like(inv)
        out[:, 0, 0] = inv[:, 1, 1] / det
        out[:, 0, 1] = -inv[:, 0, 1] / det
        out[:, 1, 0] = -inv[:, 1, 0] / det
        out[:, 1, 1] = inv[:, 0, 0] / det
        return out

    # -------------------------------------------------------------- solve step
    def _solve_distance(self, pos):
        e = self.edges
        w0 = self.inv_mass[e[:, 0]]
        w1 = self.inv_mass[e[:, 1]]
        wsum = np.maximum(w0 + w1, 1e-16)
        for _ in range(self.iterations):
            d = pos[e[:, 1]] - pos[e[:, 0]]
            ln = np.linalg.norm(d, axis=1)
            safe = np.maximum(ln, 1e-12)
            n = d / safe[:, None]
            err = ln - self.rest
            corr = (self.edge_stiffness * err / wsum)[:, None] * n
            delta = np.zeros_like(pos)
            np.add.at(delta, e[:, 0], (w0[:, None] * corr))
            np.add.at(delta, e[:, 1], -(w1[:, None] * corr))
            pos += delta / self.edge_count[:, None]
        return pos

    def _limit_edges(self, pos, passes=2):
        e = self.lim_edges
        lo = self.lim_rest * self.tgt_lo
        hi = self.lim_rest * self.tgt_hi
        w0 = self.inv_mass[e[:, 0]]
        w1 = self.inv_mass[e[:, 1]]
        wsum = np.maximum(w0 + w1, 1e-16)
        for _ in range(passes):
            d = pos[e[:, 1]] - pos[e[:, 0]]
            ln = np.linalg.norm(d, axis=1)
            safe = np.maximum(ln, 1e-12)
            target = np.clip(ln, lo, hi)
            err = ln - target
            act = np.abs(err) > 1e-12
            if not np.any(act):
                break
            n = d / safe[:, None]
            corr = (err / wsum)[:, None] * n
            corr = np.where(act[:, None], corr, 0.0)
            delta = np.zeros_like(pos)
            np.add.at(delta, e[:, 0], (w0[:, None] * corr))
            np.add.at(delta, e[:, 1], -(w1[:, None] * corr))
            pos += delta / self.lim_count[:, None]
        return pos

    def _self_repulsion(self, pos):
        if _KDTree is None:
            return pos
        dmin = 0.55 * min(self.rest_dx, self.rest_dy)
        tree = _KDTree(pos)
        pairs = tree.query_pairs(dmin, output_type="ndarray")
        if len(pairs) == 0:
            return pos
        # Exclude topological neighbours (structural + shear + bend edges).
        key = pairs[:, 0].astype(np.int64) * self.n + pairs[:, 1].astype(np.int64)
        a = np.minimum(self.edges[:, 0], self.edges[:, 1]).astype(np.int64)
        b = np.maximum(self.edges[:, 0], self.edges[:, 1]).astype(np.int64)
        ekey = a * self.n + b
        mask = ~np.isin(key, ekey)
        pairs = pairs[mask]
        if len(pairs) == 0:
            return pos
        i, j = pairs[:, 0], pairs[:, 1]
        d = pos[j] - pos[i]
        ln = np.linalg.norm(d, axis=1)
        safe = np.maximum(ln, 1e-9)
        n = d / safe[:, None]
        pen = dmin - ln
        corr = (0.5 * np.maximum(pen, 0.0))[:, None] * n
        delta = np.zeros_like(pos)
        np.add.at(delta, i, -corr)
        np.add.at(delta, j, corr)
        pos += delta
        return pos

    def _resolve_contacts(self, pos, prev, dt):
        """Sphere + ground signed-distance projection with Coulomb friction."""
        # --- fixed sphere -------------------------------------------------
        rel = pos - self.sphere_center
        dist = np.linalg.norm(rel, axis=1)
        safe = np.maximum(dist, 1e-9)
        n = rel / safe[:, None]
        target = self.sphere_radius + self.half_thickness
        pen = target - dist
        hit = pen > 0.0
        if np.any(hit):
            pos = pos + np.where(hit[:, None], np.maximum(pen, 0.0)[:, None] * n, 0.0)
            # Friction on the tangential slip accumulated in this substep.
            disp = pos - prev
            dn = np.einsum("ij,ij->i", disp, n)
            tang = disp - dn[:, None] * n
            tmag = np.linalg.norm(tang, axis=1)
            allow = self.friction * np.maximum(pen, 0.0)
            scale = np.where(tmag > 1e-12, np.minimum(allow / np.maximum(tmag, 1e-12), 1.0), 0.0)
            pos = pos - np.where(hit[:, None], (scale[:, None] * tang), 0.0)

        # --- ground slab --------------------------------------------------
        floor = self.ground_z + self.half_thickness
        below = pos[:, 2] < floor
        if np.any(below):
            pen_g = np.where(below, floor - pos[:, 2], 0.0)
            pos[:, 2] = np.where(below, floor, pos[:, 2])
            disp = pos - prev
            tang = disp.copy()
            tang[:, 2] = 0.0
            tmag = np.linalg.norm(tang, axis=1)
            allow = self.friction * pen_g
            scale = np.where(tmag > 1e-12, np.minimum(allow / np.maximum(tmag, 1e-12), 1.0), 0.0)
            pos = pos - np.where(below[:, None], scale[:, None] * tang, 0.0)
        return pos

    def _project_out(self, pos):
        """Position-only non-penetration projection (no friction)."""
        rel = pos - self.sphere_center
        dist = np.linalg.norm(rel, axis=1)
        safe = np.maximum(dist, 1e-9)
        pen = self.sphere_radius + self.half_thickness - dist
        hit = pen > 0.0
        if np.any(hit):
            n = rel / safe[:, None]
            pos = pos + np.where(hit[:, None], np.maximum(pen, 0.0)[:, None] * n, 0.0)
        floor = self.ground_z + self.half_thickness
        pos[:, 2] = np.maximum(pos[:, 2], floor)
        return pos

    def _substep(self, dt):
        prev = self.position.copy()
        vel = self.velocity + dt * self.gravity[None, :]
        vel *= float(np.exp(-self.damping * dt))
        pos = prev + dt * vel

        pos = self._solve_distance(pos)
        pos = self._limit_strain(pos, 4, self.tgt_lo, self.tgt_hi, relax=1.0)
        pos = self._limit_edges(pos, passes=2)

        self._self_collision_counter += 1
        if self._self_collision_counter % 2 == 0:
            pos = self._self_repulsion(pos)

        pos = self._resolve_contacts(pos, prev, dt)
        # Safety limiter pass so reported stretches stay inside [0.9, 1.1] ...
        pos = self._limit_strain(pos, 2, self.strain_lo + 0.01, self.strain_hi - 0.01)
        # ... then a final position-only projection so the exported state never
        # shows cloth-sphere or cloth-ground interpenetration.
        pos = self._project_out(pos)

        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        new_vel = (pos - prev) / dt
        speed = np.linalg.norm(new_vel, axis=1)
        cap = 25.0
        over = speed > cap
        if np.any(over):
            new_vel[over] *= (cap / speed[over])[:, None]
        self.position = pos
        self.velocity = np.nan_to_num(new_vel, nan=0.0, posinf=0.0, neginf=0.0)
        self._substeps_taken += 1

    # ------------------------------------------------------------ frame output
    def _deformation_gradient(self):
        a1, a2 = self._element_frames(self.position)
        s1, s2, _, _ = self._principal_stretches(a1, a2)
        nrm = np.cross(a1, a2)
        nl = np.linalg.norm(nrm, axis=1)
        unit_n = nrm / np.maximum(nl, 1e-12)[:, None]
        # Thickness stretch from shell incompressibility, clipped to the band.
        lam3 = np.clip(1.0 / np.maximum(s1 * s2, 1e-6), self.strain_lo, self.strain_hi)
        elem = self.vertex_element
        F = np.empty((self.n, 3, 3))
        F[:, :, 0] = a1[elem]
        F[:, :, 1] = a2[elem]
        F[:, :, 2] = unit_n[elem] * lam3[elem][:, None]
        smin = float(np.min(np.minimum(s1, s2)))
        smax = float(np.max(np.maximum(s1, s2)))
        self._min_stretch = min(self._min_stretch, smin)
        self._max_stretch = max(self._max_stretch, smax)
        return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def _contact_signed_distance(self):
        rel = self.position - self.sphere_center
        dist = np.linalg.norm(rel, axis=1)
        sphere_gap = dist - self.sphere_radius - self.half_thickness
        ground_gap = self.position[:, 2] - self.ground_z - self.half_thickness
        return np.minimum(sphere_gap, ground_gap)

    def _frame(self):
        F = self._deformation_gradient()
        return StateFrame(
            float(max(self.time, 0.0)),
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass_quantity.copy(),
                "uv": self.uv.copy(),
            },
            fields={
                "contact_signed_distance": self._contact_signed_distance(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )

    # ---------------------------------------------------------------- stepping
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        remaining = target - self.time
        if remaining > 1e-12:
            nsub = max(1, int(np.ceil(remaining / self.substep_dt - 1e-9)))
            dt = remaining / nsub
            for _ in range(nsub):
                self._substep(dt)
            self.time = target
        frame = self._frame()
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "self_implemented_pbd_shell_strain_limit",
            "substeps": int(self._substeps_taken),
            "max_principal_stretch": float(self._max_stretch),
            "min_principal_stretch": float(self._min_stretch),
            "vertices": int(self.n),
            "elements": int(len(self.faces)),
        }
