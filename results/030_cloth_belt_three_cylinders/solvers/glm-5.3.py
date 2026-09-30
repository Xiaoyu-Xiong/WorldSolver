"""Custom physics solver for scene 030_cloth_belt_three_cylinders.

The belt is a closed cloth strip (120 x 9 node grid, topology from the frozen
asset) wrapped around three prescribed kinematic cylinders.  The solver steps
the belt nodes with a semi-implicit (symplectic) Euler scheme:

* elasticity: a spring network on the mesh edges (structural ring/width edges
  plus weaker shear diagonals) with rest lengths shortened by the configured
  prestrain, giving an axially stiff, tensionable cloth belt;
* body forces: gravity from the frozen config plus mild viscous drag and
  along-edge damping so the belt settles without chatter;
* contact: node-vs-cylinder signed distance in the xz plane (cylinder axes are
  along y).  Penetrating nodes are projected back to the contact surface and
  inward relative normal velocity is removed (non-penetrating, energy
  dissipating, unconditionally stable - no penalty stiffness);
* friction: impulse-based Coulomb friction.  The normal impulse of the
  velocity clamp bounds a tangential stick impulse; when sticking is not
  possible the node slips with a friction impulse of magnitude mu * Jn.  The
  rotating outer cylinders therefore transport belt material through friction,
  and the belt reverses when the drive reverses.

The upper cylinder rises with a C1 smoothstep during the lift phase, which
stretches the closed belt into a taut triangular loop.  The outer cylinders
follow a C1 angular profile that completes exactly one forward revolution,
pauses, and reverses.  Both outer cylinders are driven with the same world-y
spin so their surface velocities transport the belt consistently around the
loop (the belt winds with the same sense about every cylinder).

No external physics engine is used; only NumPy/SciPy.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from worldsolver.contracts import StateFrame


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _smoothstep_d(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 6.0 * u * (1.0 - u)


def _quat_multiply(a, b):
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


class BenchmarkSolver:
    """Semi-implicit Euler cloth-belt solver with frictional cylinder contact."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        body = np.asarray(frame0.body_id, dtype=int)
        belt_mask = body == 0
        n = int(belt_mask.sum())
        if n <= 0 or not belt_mask[:n].all():
            raise ValueError("unexpected state layout: belt nodes must come first")
        self.n = n
        self.x = pos0[:n].copy()
        self.x0 = self.x.copy()
        self.v = np.zeros_like(self.x)
        self.t = 0.0
        self._substeps = 0

        belt = cfg["belt"]
        cyl = cfg["cylinders"]
        tl = cfg["timeline"]
        self.prestrain = float(belt.get("prestrain", 0.0))
        self.mu = float(cfg["friction"]["coefficient"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.Rc = float(cyl["radius"]) + float(belt.get("centerline_clearance", 0.0))
        self.total_mass = float(belt["total_mass"])
        self.m_node = self.total_mass / n
        self.W = int(belt.get("width_nodes", 9))
        if n % self.W:
            raise ValueError("belt node count is not divisible by width_nodes")
        self.ring = n // self.W

        self.lift = [float(t) for t in tl["lift"]]
        self.fwd = [float(t) for t in tl["forward"]]
        self.rev = [float(t) for t in tl["reverse"]]
        self.revolutions = float(cyl.get("outer_drive_revolutions", 1.0))
        self.cyl_xz0 = np.asarray(cyl["initial_centers_xz"], dtype=float)
        self.cyl_xz1 = np.asarray(cyl["final_centers_xz"], dtype=float)

        # ---- mesh edges (structural + shear spring network) ----
        faces = np.asarray(scene.metadata()["topology"]["faces"], dtype=np.int64)
        self.faces = faces
        e = np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
        )
        e = np.unique(e, axis=0)
        self.ei = np.ascontiguousarray(e[:, 0])
        self.ej = np.ascontiguousarray(e[:, 1])
        E = len(e)

        ia = self.ei // self.W
        ja = self.ei % self.W
        ib = self.ej // self.W
        jb = self.ej % self.W
        diagonal = (ia != ib) & (ja != jb)

        L_geom = np.linalg.norm(self.x[self.ej] - self.x[self.ei], axis=1)
        self.L0 = L_geom / (1.0 + self.prestrain)
        EA = 150.0  # belt axial stiffness [N]
        shear_fraction = 0.25
        self.k = np.where(diagonal, shear_fraction * EA, EA) / self.L0
        self.cd = 0.5  # along-edge damping [N s/m]
        self.drag = 2.0  # isotropic drag [1/s]
        self.dt_max = 5.0e-5  # substep ceiling [s]
        self.vmax = 30.0  # safety speed cap [m/s]

        # signed incidence for fast vectorized force assembly
        rows = np.concatenate([np.arange(E), np.arange(E)])
        cols = np.concatenate([self.ei, self.ej])
        vals = np.concatenate([-np.ones(E), np.ones(E)])
        self.B = sp.csr_matrix((vals, (rows, cols)), shape=(E, n))
        self.A = (-self.B.T).tocsr()

        # ---- material (rest) coordinates for the deformation gradient ----
        mid = self.x0[np.arange(self.ring) * self.W + self.W // 2]
        seg = np.linalg.norm(np.roll(mid, -1, axis=0) - mid, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        s_rest = s / (1.0 + self.prestrain)
        self.mat = np.zeros((n, 3))
        self.mat[:, 0] = s_rest[np.arange(n) // self.W]
        self.mat[:, 1] = self.x0[:, 1] / (1.0 + self.prestrain)

        self.uv = np.asarray(frame0.attributes["uv"], dtype=float).copy()
        self.body_id = body.copy()
        self.qbase = np.array([0.707106781, 0.0, 0.0, 0.707106781])
        self._initialized = True

    # ------------------------------------------------------- kinematic drivers
    def _cylinder_state(self, t):
        """Centers, center velocities, world-y spin rates and spin angles."""
        a, b = self.lift
        if t <= a:
            u = 0.0
        elif t >= b:
            u = 1.0
        else:
            u = (t - a) / (b - a)
        z_start = self.cyl_xz0[2, 1]
        z_end = self.cyl_xz1[2, 1]
        z_top = z_start + (z_end - z_start) * _smoothstep(u)
        vz_top = (z_end - z_start) * _smoothstep_d(u) / max(b - a, 1e-12)

        fa, fb = self.fwd
        ra, rb = self.rev
        two_pi = 2.0 * np.pi
        amp = two_pi * self.revolutions
        if t < fa:
            theta, omega = 0.0, 0.0
        elif t < fb:
            u = (t - fa) / (fb - fa)
            theta = amp * (u - np.sin(two_pi * u) / two_pi)
            omega = amp * (1.0 - np.cos(two_pi * u)) / (fb - fa)
        elif t < ra:
            theta, omega = amp, 0.0
        elif t < rb:
            u = (t - ra) / (rb - ra)
            theta = amp * (1.0 - (u - np.sin(two_pi * u) / two_pi))
            omega = -amp * (1.0 - np.cos(two_pi * u)) / (rb - ra)
        else:
            theta, omega = 0.0, 0.0

        centers = np.array(
            [
                [self.cyl_xz0[0, 0], 0.0, self.cyl_xz0[0, 1]],
                [self.cyl_xz0[1, 0], 0.0, self.cyl_xz0[1, 1]],
                [0.0, 0.0, z_top],
            ]
        )
        vels = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, vz_top]]
        )
        # Equal world-y spin on both outer cylinders: the belt winds with the
        # same sense about every cylinder, so equal spins transport it around
        # the closed loop consistently (forward, then reversed).
        omegas = np.array([omega, omega, 0.0])
        spins = np.array([theta, theta, 0.0])
        return centers, vels, omegas, spins

    # ---------------------------------------------------------------- stepping
    def _substep(self, t: float, h: float) -> None:
        x, v = self.x, self.v
        centers, vels, omegas, _spins = self._cylinder_state(t)

        # --- elastic forces on the spring network ---
        d = x[self.ej] - x[self.ei]
        L = np.linalg.norm(d, axis=1)
        Lsafe = np.maximum(L, 1e-12)
        direction = d / Lsafe[:, None]
        dv = self.B @ v  # (E,3) relative endpoint velocities
        vrel = np.einsum("ij,ij->i", dv, direction)
        f = self.k * (L - self.L0) + self.cd * vrel
        force = self.A @ (f[:, None] * direction)  # (n,3)

        v += h * (self.gravity[None, :] + force * (1.0 / self.m_node))
        v *= 1.0 / (1.0 + self.drag * h)
        x += h * v

        # --- frictional contact with the three cylinders ---
        for c in range(3):
            dx = x - centers[c]
            r = np.sqrt(dx[:, 0] ** 2 + dx[:, 2] ** 2)
            gap = r - self.Rc
            idx = np.flatnonzero(gap < 0.0)
            if idx.size == 0:
                continue
            ri = np.maximum(r[idx], 1e-9)
            zeros = np.zeros(idx.size)
            normal = np.stack([dx[idx, 0] / ri, zeros, dx[idx, 2] / ri], axis=1)
            # surface velocity: omega (about +y) x offset, plus center motion
            vsurf = omegas[c] * np.stack(
                [dx[idx, 2], zeros, -dx[idx, 0]], axis=1
            ) + vels[c]
            vi = v[idx].copy()
            vrel_vec = vi - vsurf
            vn = np.einsum("ij,ij->i", vrel_vec, normal)
            # normal impulse available for friction (removed approach momentum)
            Jn = self.m_node * np.maximum(0.0, -vn)
            approaching = vn < 0.0
            if np.any(approaching):
                vi[approaching] -= vn[approaching][:, None] * normal[approaching]
            # tangential (Coulomb) friction impulse
            vrel_vec = vi - vsurf
            vn2 = np.einsum("ij,ij->i", vrel_vec, normal)
            vt = vrel_vec - vn2[:, None] * normal
            vts = np.linalg.norm(vt, axis=1)
            Jf = np.minimum(self.mu * Jn, self.m_node * vts)
            scale = np.where(vts > 1e-12, Jf / np.maximum(vts, 1e-12), 0.0)
            vi -= scale[:, None] * vt
            v[idx] = vi
            # position-level non-penetration projection
            push = -gap[idx]
            x[idx, 0] += push * normal[:, 0]
            x[idx, 2] += push * normal[:, 2]

        # --- determinism / safety guards ---
        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        speed2 = np.einsum("ij,ij->i", v, v)
        fast = speed2 > self.vmax * self.vmax
        if fast.any():
            v[fast] *= (self.vmax / np.sqrt(speed2[fast]))[:, None]
        self._substeps += 1

    # ------------------------------------------------------------------ frames
    def advance_to(self, time):
        if not self._initialized:
            raise RuntimeError("initialize() must be called before advance_to()")
        time = float(time)
        if time > self.t + 1e-12:
            span = time - self.t
            steps = int(np.ceil(span / self.dt_max - 1e-9))
            steps = max(steps, 1)
            h = span / steps
            for i in range(steps):
                self._substep(self.t + i * h, h)
            self.t = time
        elif time < self.t - 1e-9:
            raise ValueError("advance_to() requires non-decreasing times")
        return self._frame(time)

    def _deformation_gradient(self):
        """Per-node 3x3 deformation gradient (area-averaged over faces)."""
        faces = self.faces
        p = self.x[faces]
        c1 = p[:, 1] - p[:, 0]
        c2 = p[:, 2] - p[:, 0]
        cr = np.cross(c1, c2)
        cn = np.maximum(np.linalg.norm(cr, axis=1), 1e-14)
        c3 = cr / cn[:, None]
        m1 = self.mat[faces[:, 1]] - self.mat[faces[:, 0]]
        m2 = self.mat[faces[:, 2]] - self.mat[faces[:, 0]]
        F_count = len(faces)
        M = np.zeros((F_count, 3, 3))
        M[:, :, 0] = m1
        M[:, :, 1] = m2
        M[:, 2, 2] = 1.0
        C = np.zeros_like(M)
        C[:, :, 0] = c1
        C[:, :, 1] = c2
        C[:, :, 2] = c3
        F_face = np.matmul(C, np.linalg.inv(M))
        F_node = np.zeros((self.n, 3, 3))
        counts = np.zeros(self.n)
        fidx = faces.reshape(-1)
        np.add.at(F_node, fidx, np.repeat(F_face, 3, axis=0))
        np.add.at(counts, fidx, 1)
        F_node /= np.maximum(counts, 1)[:, None, None]
        # rigid cylinder samples carry the identity map
        return np.concatenate([F_node, np.tile(np.eye(3), (3, 1, 1))], axis=0)

    def _frame(self, time: float) -> StateFrame:
        centers, vels, omegas, spins = self._cylinder_state(time)
        n = self.n
        N = n + 3
        position = np.vstack([self.x, centers])
        velocity = np.vstack([self.v, vels])

        orientation = np.zeros((N, 4))
        orientation[:, 3] = 1.0
        for k in range(3):
            half = 0.5 * spins[k]
            qy = np.array([0.0, np.sin(half), 0.0, np.cos(half)])
            orientation[n + k] = _quat_multiply(qy, self.qbase)
        angular_velocity = np.zeros((N, 3))
        angular_velocity[n:, 1] = omegas

        mass = np.concatenate([np.full(n, self.m_node), np.zeros(3)])

        deformation = self._deformation_gradient()
        np.nan_to_num(deformation, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        gaps = None
        for c in range(3):
            dxc = self.x - centers[c]
            g = np.sqrt(dxc[:, 0] ** 2 + dxc[:, 2] ** 2) - self.Rc
            gaps = g if gaps is None else np.minimum(gaps, g)
        near = np.flatnonzero(gaps < 0.05)
        contact_gap = gaps[near] if near.size else np.zeros(1)

        frame = StateFrame(
            float(time),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": mass.copy(),
                "deformation_gradient": deformation,
            },
            fields={
                "contact_signed_distance": contact_gap,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "contact_length_scale": float(self.cfg["belt"]["thickness"]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "method": (
                "semi-implicit Euler cloth solver with spring-network elasticity, "
                "projection contact and impulse Coulomb friction on kinematic "
                "cylinders"
            ),
            "substeps": int(self._substeps),
            "final_time": float(self.t),
            "belt_mass": float(self.total_mass),
            "belt_nodes": int(self.n),
            "seed": int(self.seed),
        }
