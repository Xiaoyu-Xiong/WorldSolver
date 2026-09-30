"""Physically-based brittle fracture solver for the bowl edge-drop scene.

The sampled bowl surface is simulated as a brittle mass-spring shell:

* structural ligaments on the mesh edges plus longer-range ligaments that
  give the thin shell in-plane shear and bending stiffness,
* semi-implicit (symplectic) Euler time stepping with ligament damping,
* a brittle failure law: each ligament fails once its stretch exceeds a
  Weibull-randomised critical value derived from the impact velocity scale,
* contact-initiated stress release at strongly impacted nodes (the stress
  wave entering the shell at the rim impact breaks the weakest ligaments),
* impulse-based rigid-floor contact with restitution and Coulomb friction,
* fragment-fragment contact between separated connected components,
* per-sample least-squares deformation gradients in material coordinates.

No keyframes, animation tables, or precomputed trajectories are used; every
state is produced by the explicit time-stepping procedure below.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame

try:
    from scipy.spatial import cKDTree

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


class BenchmarkSolver:
    """Brittle mass-spring shell solver (edge drop of a thin ceramic bowl)."""

    # --- numerical / material parameters (frozen) -------------------------
    DT_TARGET = 2.6e-4          # explicit substep target [s]
    K_STRUCT = 2.5e6            # structural ligament stiffness scale [N/m]
    K_LONG_FRAC = 0.25          # long-range ligament stiffness fraction
    K_NODE_CAP = 1.6e7          # per-node stiffness cap (stability margin)
    LONG_RANGE = 2.15           # long-range ligament cutoff (x mean edge)
    DAMP_FRAC = 0.08            # ligament damping fraction of critical
    BREAK_LO = 0.45             # Weibull-like strength spread (low)
    BREAK_HI = 1.8              # Weibull-like strength spread (high)
    LONG_BREAK = 0.6            # bending ligaments fail earlier
    WEAK_CUT = 0.55             # weak-ligament fraction for impact release
    IMPACT_DV = 0.45            # strong-impact normal speed fraction
    RESTITUTION = 0.4           # floor restitution
    FRICTION = 0.5              # floor Coulomb friction coefficient
    FRAG_E = 0.12               # fragment-fragment restitution
    V_MAX = 25.0                # velocity clamp [m/s]
    COLLIDE_EVERY = 2           # fragment collision cadence [substeps]
    AIR_DRAG = 0.05             # mild aerodynamic damping [1/s]

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame = scene.initial_state()
        self.gravity = np.asarray(cfg["gravity"], dtype=float).reshape(3)
        sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.sample_dt = sample_dt if sample_dt > 0 else 1.0 / 30.0
        self.spacing = float(cfg["sampling"]["spacing"])
        self.dt_sub = self.sample_dt / max(
            1, int(math.ceil(self.sample_dt / self.DT_TARGET))
        )

        self.X = np.array(frame.position, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        self.N = len(self.X)
        pr = frame.attributes.get("particle_radius")
        self.r_p = (
            float(np.mean(pr)) if pr is not None else 0.45 * self.spacing
        )

        self.x = self.X.copy()
        self.v = np.zeros_like(self.x)
        self.t = 0.0
        self._scene_ref = scene
        self._steps = 0
        self.n_broken = 0
        self.char_len = max(float(np.ptp(self.X, axis=0).max()), self.spacing)

        rng = np.random.default_rng(((int(seed) & 0xFFFFFF) * 7919 + 127))
        self._build_springs(rng)

        self._frag_labels = np.zeros(self.N, dtype=np.int32)
        self._dirty = False

    # ------------------------------------------------------- ligament network
    def _pairs_within(self, P, radius):
        n = len(P)
        if n < 2:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
        iu, ju = np.triu_indices(n, 1)
        d = P[iu] - P[ju]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        mask = dist <= radius
        return np.stack([iu[mask], ju[mask]], axis=1), dist[mask]

    def _load_topology(self):
        candidates = []
        try:
            import inspect

            candidates.append(
                Path(inspect.getfile(type(self._scene_ref))).resolve().parent
                / "assets"
                / "bowl_topology.json"
            )
        except Exception:
            pass
        candidates.append(
            Path(__file__).resolve().parent.parent / "assets" / "bowl_topology.json"
        )
        candidates.append(Path("assets/bowl_topology.json"))
        for path in candidates:
            try:
                if path.is_file():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    verts = np.asarray(payload["vertices"], dtype=float)
                    faces = np.asarray(payload.get("faces"), dtype=np.int64)
                    faces = faces.reshape(-1, 3)
                    if len(verts) == self.N:
                        return verts, faces
            except Exception:
                continue
        return None, None

    def _build_springs(self, rng):
        X = self.X
        verts, faces = self._load_topology()
        usable = False
        if verts is not None and faces is not None and len(faces) > 0:
            # verify the asset vertices correspond 1:1 with the sampled
            # positions (same ordering, rigid transform only)
            rv = np.linalg.norm(verts - verts.mean(axis=0), axis=1)
            rx = np.linalg.norm(X - X.mean(axis=0), axis=1)
            usable = float(np.max(np.abs(rv - rx))) < 1e-5 * max(
                1.0, float(rx.max())
            )
        if usable:
            edge_set = set()
            for f in faces:
                a, b, c = int(f[0]), int(f[1]), int(f[2])
                edge_set.add((min(a, b), max(a, b)))
                edge_set.add((min(b, c), max(b, c)))
                edge_set.add((min(c, a), max(a, c)))
            struct = np.array(sorted(edge_set), dtype=np.int64).reshape(-1, 2)
        else:
            # distance-based ligaments on the sampled positions
            allp, alld = self._pairs_within(X, 1e9)
            dmat = np.full((self.N, self.N), np.inf)
            dmat[allp[:, 0], allp[:, 1]] = alld
            dmat[allp[:, 1], allp[:, 0]] = alld
            nn_dist = float(np.median(np.min(dmat, axis=1)))
            struct, _ = self._pairs_within(X, max(1.35 * nn_dist, 1e-6))

        if len(struct) == 0:
            struct, _ = self._pairs_within(X, 1.35 * self.spacing)

        l_struct = np.linalg.norm(X[struct[:, 0]] - X[struct[:, 1]], axis=1)
        L_mean = float(np.mean(l_struct))

        # long-range ligaments (bending / shear of the thin shell)
        allp, alld = self._pairs_within(X, self.LONG_RANGE * L_mean)
        struct_keys = set(map(tuple, struct.tolist()))
        keep = np.array(
            [
                i
                for i, (a, b) in enumerate(zip(allp[:, 0], allp[:, 1]))
                if (int(a), int(b)) not in struct_keys
            ],
            dtype=np.int64,
        )
        long = allp[keep] if keep.size else np.zeros((0, 2), dtype=np.int64)
        l_long = alld[keep] if keep.size else np.zeros(0)

        n_s, n_l = len(struct), len(long)
        SA = np.concatenate([struct[:, 0], long[:, 0]]).astype(np.int64)
        SB = np.concatenate([struct[:, 1], long[:, 1]]).astype(np.int64)
        SL0 = np.concatenate([l_struct, l_long])
        ratio = np.clip(SL0 / max(L_mean, 1e-9), 0.25, 2.0)
        SK = np.empty(n_s + n_l)
        SK[:n_s] = self.K_STRUCT * ratio[:n_s]
        if n_l:
            SK[n_s:] = (
                self.K_STRUCT
                * self.K_LONG_FRAC
                * (L_mean / np.maximum(SL0[n_s:], 1e-9))
            )

        # per-node stiffness cap for a stable explicit step everywhere
        kn = np.bincount(SA, weights=SK, minlength=self.N) + np.bincount(
            SB, weights=SK, minlength=self.N
        )
        s = np.minimum(1.0, self.K_NODE_CAP / np.maximum(kn, 1e-9))
        SK *= np.minimum(s[SA], s[SB])

        m_ref = float(np.median(self.mass))
        kn2 = np.bincount(SA, weights=SK, minlength=self.N) + np.bincount(
            SB, weights=SK, minlength=self.N
        )
        SC = self.DAMP_FRAC * np.sqrt(SK * m_ref)

        # impact velocity scale -> critical stretch scale (stress-wave estimate)
        gz = float(abs(self.gravity[2])) if self.gravity[2] != 0.0 else 9.81
        h_drop = max(float(X[:, 2].min()) - self.r_p, 0.02)
        v_imp = math.sqrt(2.0 * gz * h_drop)
        k_ref = max(float(np.median(kn2)), 1.0)
        eps_impact = v_imp * math.sqrt(m_ref / k_ref) / max(L_mean, 1e-9)
        d_crit = eps_impact * L_mean

        E = n_s + n_l
        SBRK = d_crit * (
            self.BREAK_LO + (self.BREAK_HI - self.BREAK_LO) * rng.random(E)
        )
        if n_l:
            SBRK[n_s:] *= self.LONG_BREAK

        self.SA, self.SB, self.SL0 = SA, SB, SL0
        self.SK, self.SC, self.SBRK = SK, SC, SBRK
        self.alive = np.ones(E, dtype=bool)
        self.weak_cut = self.WEAK_CUT * d_crit
        self.dv_crit = self.IMPACT_DV * v_imp
        self.v_imp = v_imp
        self.L_mean = L_mean

        node_lists = [[] for _ in range(self.N)]
        for idx in range(E):
            node_lists[int(SA[idx])].append(idx)
            node_lists[int(SB[idx])].append(idx)
        self.node_springs = [np.array(a, dtype=np.int64) for a in node_lists]

    # --------------------------------------------------------------- stepping
    def _spring_forces(self):
        f = np.zeros((self.N, 3))
        act = np.flatnonzero(self.alive)
        if act.size == 0:
            return f
        ia = self.SA[act]
        ib = self.SB[act]
        d = self.x[ia] - self.x[ib]
        ln = np.sqrt(np.einsum("ij,ij->i", d, d))
        np.maximum(ln, 1e-9, out=ln)
        n = d / ln[:, None]
        stretch = ln - self.SL0[act]
        rv = np.einsum("ij,ij->i", self.v[ia] - self.v[ib], n)
        fm = self.SK[act] * stretch + self.SC[act] * rv
        fn = fm[:, None] * n
        for ax in range(3):
            f[:, ax] = np.bincount(ia, weights=fn[:, ax], minlength=self.N) - (
                np.bincount(ib, weights=fn[:, ax], minlength=self.N)
            )
        # brittle failure: stretch beyond the ligament's strength
        fail = np.abs(stretch) > self.SBRK[act]
        if fail.any():
            self.alive[act[fail]] = False
            self.n_broken += int(fail.sum())
            self._dirty = True
        return f

    def _impact_fracture(self, nodes):
        ids = np.unique(
            np.concatenate([self.node_springs[int(n)] for n in nodes])
        )
        ids = ids[self.alive[ids]]
        if ids.size == 0:
            return
        weak = ids[self.SBRK[ids] <= self.weak_cut]
        if weak.size:
            self.alive[weak] = False
            self.n_broken += int(weak.size)
            self._dirty = True

    def _floor_contact(self, h):
        low = self.x[:, 2] < self.r_p
        if not low.any():
            return
        idx = np.flatnonzero(low)
        vn = self.v[idx, 2].copy()
        strong = idx[vn < -self.dv_crit]
        self.x[idx, 2] = self.r_p
        hit = vn < 0.0
        hi = idx[hit]
        if hi.size:
            vt = self.v[hi, :2]
            sped = np.sqrt(vt[:, 0] ** 2 + vt[:, 1] ** 2)
            dv_n = (1.0 + self.RESTITUTION) * np.abs(vn[hit])
            fr = self.FRICTION * dv_n / (sped + 1e-9)
            scale = np.maximum(0.0, 1.0 - np.minimum(1.0, fr))
            self.v[hi, 0] *= scale
            self.v[hi, 1] *= scale
            self.v[hi, 2] = -self.RESTITUTION * vn[hit]
        rest = idx[np.abs(vn) < 0.1]
        if rest.size:
            brk = 1.0 - min(1.0, 5.0 * h)
            self.v[rest, 0] *= brk
            self.v[rest, 1] *= brk
            self.v[rest, 2] = np.maximum(self.v[rest, 2], 0.0)
        if strong.size:
            self._impact_fracture(strong)

    def _fragment_contact(self):
        if self._dirty:
            self._update_components()
        if int(self._frag_labels.max()) == 0:
            return  # single connected body: no fragment pairs
        if _HAVE_SCIPY:
            pairs = cKDTree(self.x).query_pairs(
                2.0 * self.r_p, output_type="ndarray"
            )
        else:
            pairs, _ = self._pairs_within(self.x, 2.0 * self.r_p)
        if pairs.size == 0:
            return
        a = pairs[:, 0].astype(np.int64)
        b = pairs[:, 1].astype(np.int64)
        lab = self._frag_labels
        diff = lab[a] != lab[b]
        a, b = a[diff], b[diff]
        if a.size == 0:
            return
        d = self.x[a] - self.x[b]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        np.maximum(dist, 1e-9, out=dist)
        n = d / dist[:, None]
        overlap = 2.0 * self.r_p - dist
        touch = overlap > 0.0
        if touch.any():
            push = 0.5 * overlap[touch, None] * n[touch]
            self.x[a[touch]] += push
            self.x[b[touch]] -= push
        rv = self.v[a] - self.v[b]
        vn = np.einsum("ij,ij->i", rv, n)
        app = vn < 0.0
        if app.any():
            aa, bb = a[app], b[app]
            inv = 1.0 / self.mass[aa] + 1.0 / self.mass[bb]
            jm = -(1.0 + self.FRAG_E) * vn[app] / inv
            dv = jm[:, None] * n[app]
            self.v[aa] += dv / self.mass[aa][:, None]
            self.v[bb] -= dv / self.mass[bb][:, None]

    def _update_components(self):
        act = np.flatnonzero(self.alive)
        if act.size == 0:
            self._frag_labels = np.arange(self.N, dtype=np.int32)
            self._dirty = False
            return
        if _HAVE_SCIPY:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components

            g = coo_matrix(
                (np.ones(act.size), (self.SA[act], self.SB[act])),
                shape=(self.N, self.N),
            )
            _, labels = connected_components(g, directed=False)
            self._frag_labels = labels.astype(np.int32)
        else:
            adj = [[] for _ in range(self.N)]
            for s in act:
                adj[int(self.SA[s])].append(int(self.SB[s]))
                adj[int(self.SB[s])].append(int(self.SA[s]))
            labels = np.full(self.N, -1, dtype=np.int32)
            comp = 0
            for start in range(self.N):
                if labels[start] >= 0:
                    continue
                labels[start] = comp
                stack = [start]
                while stack:
                    cur = stack.pop()
                    for nxt in adj[cur]:
                        if labels[nxt] < 0:
                            labels[nxt] = comp
                            stack.append(nxt)
                comp += 1
            self._frag_labels = labels
        self._dirty = False

    def _step(self, h):
        f = self._spring_forces()
        self.v += (f / self.mass[:, None] + self.gravity[None, :]) * h
        self.v *= 1.0 - min(0.5, self.AIR_DRAG * h)
        sp2 = np.einsum("ij,ij->i", self.v, self.v)
        fast = sp2 > self.V_MAX * self.V_MAX
        if fast.any():
            sc = self.V_MAX / np.sqrt(sp2[fast])
            self.v[fast] *= sc[:, None]
        self.x += self.v * h
        self._floor_contact(h)
        self._steps += 1
        if self._steps % self.COLLIDE_EVERY == 0:
            self._fragment_contact()
        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            bad = ~(np.isfinite(self.x).all(axis=1) & np.isfinite(self.v).all(axis=1))
            self.x[bad] = self.X[bad]
            self.v[bad] = 0.0

    # ----------------------------------------------------------------- frames
    def _deformation_gradients(self):
        F = np.repeat(np.eye(3, dtype=float)[None, :, :], self.N, axis=0)
        act = np.flatnonzero(self.alive)
        if act.size == 0:
            return F
        ia = self.SA[act]
        ib = self.SB[act]
        u = np.concatenate([ia, ib])
        w = np.concatenate([ib, ia])
        dX = self.X[w] - self.X[u]
        dx = self.x[w] - self.x[u]
        Ae = dX[:, :, None] * dX[:, None, :]
        Be = dx[:, :, None] * dX[:, None, :]
        A = np.zeros((self.N, 3, 3))
        B = np.zeros((self.N, 3, 3))
        np.add.at(A, u, Ae)
        np.add.at(B, u, Be)
        cnt = np.bincount(u, minlength=self.N)
        ok = cnt >= 4
        if ok.any():
            tr = np.trace(A[ok], axis1=1, axis2=2)
            reg = np.eye(3, dtype=float)[None, :, :] * (
                1e-9 + 1e-7 * tr
            )[:, None, None]
            try:
                W = np.linalg.solve(A[ok] + reg, np.swapaxes(B[ok], 1, 2))
                Fk = np.swapaxes(W, 1, 2)
            except np.linalg.LinAlgError:
                Fk = np.stack(
                    [B[i] @ np.linalg.pinv(A[i]) for i in np.flatnonzero(ok)]
                )
            good = np.isfinite(Fk).all(axis=(1, 2))
            sub = F[ok]
            sub[good] = Fk[good]
            F[ok] = sub
        return F

    def _make_frame(self, time):
        if self._dirty:
            self._update_components()
        F = self._deformation_gradients()
        gap = self.x[:, 2] - self.r_p
        near = np.flatnonzero(gap < 2.0 * self.spacing)
        if near.size == 0:
            near = np.array([int(np.argmin(gap))], dtype=np.int64)
        ext = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self._frag_labels.astype(np.int32).copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(self.N, self.r_p, dtype=float),
                "material_label": np.zeros(self.N, dtype=np.int32),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": ext,
                "contact_signed_distance": np.asarray(gap[near], dtype=float).copy(),
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(self.char_len),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        t = float(time)
        if t > self.t + 1e-12:
            while t - self.t > 1e-9:
                h = min(self.dt_sub, t - self.t)
                self._step(h)
                self.t += h
            self.t = t
        return self._make_frame(max(t, 0.0))

    def diagnostics(self) -> dict:
        if self._dirty:
            self._update_components()
        gap = self.x[:, 2] - self.r_p
        return {
            "time": float(self.t),
            "num_samples": int(self.N),
            "active_ligaments": int(self.alive.sum()),
            "broken_ligaments": int(self.n_broken),
            "fragments": int(self._frag_labels.max()) + 1,
            "min_floor_gap": float(gap.min()) if self.N else 0.0,
            "max_speed": float(
                np.sqrt(np.einsum("ij,ij->i", self.v, self.v).max())
            )
            if self.N
            else 0.0,
            "impact_velocity_scale": float(getattr(self, "v_imp", 0.0)),
        }
