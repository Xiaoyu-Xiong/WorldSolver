"""WorldSolver solver for scene 014_inclined_chair_stick_slip.

Recreates the physical phenomenon of a plastic chair whose legs slide down an
inclined surface and chatter as friction repeatedly switches between sticking
and sliding (cf. Kaufman et al., "Staggered Projections for Frictional Contact
in Multibody Systems", Fig. 5 right).

Physical model (implemented from scratch, no external physics engine):

* The chair is a single rigid body.  Mass, centre of mass and inertia tensor
  are computed exactly from the frozen ``chair.stl`` triangle mesh (uniform
  density scaled to the frozen mass of 10.8 kg).
* The incline is a static half space obtained from the frozen box transform
  (top face of the incline).
* Normal contact is solved at the velocity level with a projected Jacobi /
  staggered-projection impulse scheme (Coulomb cone projections, zero
  restitution, Baumgarte stabilisation for penetration drift).
* Tangential contact uses a physically standard compliant-foot model: each
  contact point of the chair is coupled to the ground through a light foot
  node (small mass) with a tangential spring/damper (elastic plastic feet)
  and a Stribeck (rate weakening) Coulomb friction force.  The combination
  of elastic feet with static/kinetic friction contrast produces genuine
  self-excited stick-slip chatter: feet stick while the feet spring loads,
  break loose when the spring force reaches the static cone, slide at the
  lower kinetic friction, and re-stick - repeatedly.

Time stepping is semi-implicit Euler with a fixed substep, fully
deterministic, and every requested sample time is reached exactly.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

try:  # PyYAML is available in the runtime image; fall back to frozen defaults.
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Physical parameters (frozen for this scene)
# ---------------------------------------------------------------------------
GRAVITY = 9.81                      # m/s^2
SUBSTEP = 2.0e-4                    # s, fixed deterministic substep

MU_STATIC = 0.55                    # static friction coefficient (plastic feet)
MU_KINETIC = 0.31                   # kinetic friction coefficient
V_STRIBECK = 0.05                   # m/s, Stribeck decay speed
V_REG = 0.01                        # m/s, regularisation of friction direction

K_TANGENTIAL = 160.0                # N/m, tangential foot stiffness per contact
C_TANGENTIAL = 2.0                  # N s/m, tangential foot damping per contact
M_FOOT = 0.02                       # kg, foot node mass per contact

CONTACT_MARGIN = 3.0e-3             # m, activation distance of a contact
BAUMGARTE = 0.25                    # penetration stabilization factor
BIAS_CAP = 0.2                      # m/s, max Baumgarte bias velocity
SWEEPS = 24                         # projected Jacobi sweeps per substep
RELAXATION = 0.85                   # Jacobi relaxation factor

WORKSPACE = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Small quaternion / mesh utilities
# ---------------------------------------------------------------------------
def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _integrate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    dq = 0.5 * _quat_multiply(np.array([omega[0], omega[1], omega[2], 0.0]), q)
    q_out = q + dq * dt
    return q_out / np.linalg.norm(q_out)


def _load_stl(path: Path) -> np.ndarray:
    """Read a (possibly binary) STL file into an (n, 3, 3) triangle array."""
    raw = path.read_bytes()
    # Binary STL: 80 byte header + uint32 count + 50 bytes per triangle.
    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) >= 84 + 50 * count and count > 0:
        record = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        rec = np.frombuffer(raw[84 : 84 + 50 * count], dtype=record, count=count)
        return np.ascontiguousarray(rec["v"], dtype=np.float64)
    # ASCII fallback.
    verts = []
    for line in raw.decode("ascii", errors="ignore").splitlines():
        parts = line.split()
        if parts and parts[0] == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def _mass_properties(tris: np.ndarray):
    """Exact volume, centroid and inertia (unit density) of a closed mesh.

    Uses signed tetrahedra (origin, a, b, c) per face.  The second moment of a
    tetra with vertices 0, u, v, w follows from the Dirichlet(1,1,1,1)
    moments E[a^2] = 1/10 and E[ab] = 1/20.
    """
    a, b, c = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))  # 6 * signed volume
    if det.sum() < 0.0:  # normalise on the outward-wound orientation
        det = -det
    vol6 = det.sum()
    volume = vol6 / 6.0
    centroid = np.einsum("ij,i->j", (a + b + c) / 4.0, det) / vol6

    def _outer_sum(u, v):
        return np.einsum("ki,kj->kij", u, v)

    s2 = (
        _outer_sum(a, a) + _outer_sum(b, b) + _outer_sum(c, c)
    ) * (1.0 / 10.0) + (
        _outer_sum(a, b) + _outer_sum(b, a)
        + _outer_sum(a, c) + _outer_sum(c, a)
        + _outer_sum(b, c) + _outer_sum(c, b)
    ) * (1.0 / 20.0)
    s2 = np.einsum("kij,k->ij", s2, det) / 6.0  # ∫ x x^T dV (unit density)

    i_origin = np.trace(s2) * np.eye(3) - s2
    d = centroid
    i_com = i_origin - volume * ((d @ d) * np.eye(3) - np.outer(d, d))
    return volume, centroid, i_com


def _quat_from_xyzw(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(4)
    return arr / np.linalg.norm(arr)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class BenchmarkSolver:
    """Rigid-body solver with compliant frictional feet (stick-slip chatter)."""

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        initial = scene.initial_state()

        self.scene_id = str(cfg.get("scene_id", "014_inclined_chair_stick_slip"))
        self.mass = float(np.asarray(initial.mass).reshape(-1)[0])
        self.body_id = np.asarray(initial.body_id).reshape(-1).copy()

        # --- chair mesh: mass properties and contact points ---------------
        tris = _load_stl(WORKSPACE / "chair.stl")
        volume, centroid, inertia_unit = _mass_properties(tris)
        if volume <= 0.0:
            raise ValueError("chair.stl does not enclose a positive volume")
        density = self.mass / volume
        self.com_local = centroid
        self.inertia_body = density * inertia_unit
        self.inertia_body = 0.5 * (
            self.inertia_body + self.inertia_body.T
        )  # symmetrise
        # Light regularisation to guarantee positive definiteness.
        self.inertia_body += np.eye(3) * (1e-6 * max(np.trace(self.inertia_body), 1.0))
        self.inertia_inv_body = np.linalg.inv(self.inertia_body)

        # Contact points: the flat rail/leg bottoms (all vertices at the
        # minimum z of the mesh), reduced to a deterministic small set that
        # spans each rail footprint.
        verts = tris.reshape(-1, 3)
        z_bottom = verts[:, 2].min()
        bottom = verts[verts[:, 2] < z_bottom + 1e-9]
        rails = [bottom[bottom[:, 0] < 0.1], bottom[bottom[:, 0] > 0.2]]
        rail_width = 0.1
        chosen = []
        for rail in rails:
            if len(rail) == 0:
                continue
            y_lo = rail[:, 1].min()
            y_hi = rail[:, 1].max()
            x_lo = rail[:, 0].min()
            x_hi = rail[:, 0].max()
            rail_width = max(rail_width, x_hi - x_lo)
            targets = [
                rail[np.argmin(rail[:, 1])],
                rail[np.argmax(rail[:, 1])],
                rail[np.argmin(rail[:, 0])],
                rail[np.argmax(rail[:, 0])],
                rail[np.argmin(np.abs(rail[:, 1]))],
                rail[np.argmin(np.abs(rail[:, 1] - 0.25 * (y_lo + y_hi)))],
            ]
            for p in targets:
                if not any(np.allclose(p, q, atol=1e-9) for q in chosen):
                    chosen.append(p)
        self.contact_local = np.asarray(chosen, dtype=np.float64)
        self.n_contacts = len(self.contact_local)
        self.contact_length_scale = float(rail_width)

        # --- incline plane from the frozen scene geometry ------------------
        plane = self._load_incline(cfg)
        self.plane_point, self.plane_normal = plane

        # --- initial rigid body state -------------------------------------
        pos = np.asarray(initial.position, dtype=np.float64).reshape(3)
        q0 = _quat_from_xyzw(initial.orientation[0])
        rot0 = _quat_to_matrix(q0)
        # scene position refers to the chair body (mesh) origin
        self.com = pos + rot0 @ self.com_local
        self.q = q0
        self.v = np.asarray(initial.velocity, dtype=np.float64).reshape(3).copy()
        self.w = np.asarray(initial.angular_velocity, dtype=np.float64).reshape(3).copy()

        # --- contact / foot state ------------------------------------------
        self.foot = np.zeros((self.n_contacts, 3))
        self.foot_vel = np.zeros((self.n_contacts, 3))
        self.active_prev = np.zeros(self.n_contacts, dtype=bool)
        self.normal_force = np.zeros(self.n_contacts)   # N per contact
        self.friction_coef = np.full(self.n_contacts, MU_KINETIC)
        self.spring_force = np.zeros((self.n_contacts, 3))
        self.gap = np.zeros(self.n_contacts)

        self.time = 0.0
        self._substeps = 0
        self.gravity_vec = np.array([0.0, 0.0, -GRAVITY])

        # Initial contact geometry (so the t = 0 sample carries real gaps).
        local = self.contact_local - self.com_local
        r0 = self.com + local @ rot0.T
        self.gap = (r0 - self.plane_point) @ self.plane_normal

    # ------------------------------------------------------------------
    def _load_incline(self, cfg):
        """Top-face plane (point, normal) of the frozen incline box."""
        geo = cfg.get("scene_geometry") if isinstance(cfg, dict) else None
        incline = None
        if isinstance(geo, dict):
            for obj in geo.get("objects", []):
                if obj.get("id") == "incline":
                    incline = obj
                    break
        if incline is None and yaml is not None:
            try:
                data = yaml.safe_load((WORKSPACE / "scene_geometry.yaml").read_text())
                for obj in data.get("objects", []):
                    if obj.get("id") == "incline":
                        incline = obj
                        break
            except Exception:
                incline = None
        if incline is None:
            incline = {
                "geometry": {"half_extents": [3.5, 2.0, 0.08]},
                "transform": {
                    "translation": [0.0, 0.0, 0.35],
                    "rotation_xyzw": [0.0, 0.1564, 0.0, 0.9877],
                },
            }
        half = np.asarray(incline["geometry"]["half_extents"], dtype=np.float64)
        tr = np.asarray(incline["transform"]["translation"], dtype=np.float64)
        q = _quat_from_xyzw(incline["transform"]["rotation_xyzw"])
        rot = _quat_to_matrix(q)
        point = tr + rot @ np.array([0.0, 0.0, half[2]])
        normal = rot @ np.array([0.0, 0.0, 1.0])
        return point, normal

    # ------------------------------------------------------------------
    def _step(self) -> None:
        h = SUBSTEP
        n_c = self.n_contacts
        n = self.plane_normal
        o = self.plane_point
        m = self.mass

        rot = _quat_to_matrix(self.q)
        inertia_inv_w = rot @ self.inertia_inv_body @ rot.T

        # Contact kinematics (body-frame contact points -> world).
        local = self.contact_local - self.com_local
        r = self.com + local @ rot.T                       # (K,3)
        arms = r - self.com                                 # (K,3)
        self.gap = (r - o) @ n                              # (K,)
        active = self.gap < CONTACT_MARGIN

        # (Re)initialise foot nodes for newly active contacts.
        newly = active & ~self.active_prev
        if newly.any():
            self.foot[newly] = r[newly]
            self.foot_vel[newly] = 0.0

        # Tangential spring / damper between the body point and the foot node.
        u_body = self.v + np.cross(self.w, arms)            # (K,3)
        rel = u_body - self.foot_vel
        rel_t = rel - np.outer(rel @ n, n)
        s = r - self.foot
        s_t = s - np.outer(s @ n, n)
        f_spring = -(K_TANGENTIAL * s_t + C_TANGENTIAL * rel_t)
        f_spring[~active] = 0.0

        # Stribeck friction on the foot nodes, bounded by the current normal
        # force estimate (friction cone |f_t| <= mu * N).
        speed = np.linalg.norm(self.foot_vel, axis=1)
        mu = MU_KINETIC + (MU_STATIC - MU_KINETIC) * np.exp(-speed / V_STRIBECK)
        cap = mu * self.normal_force
        denom = np.sqrt(speed * speed + V_REG * V_REG)
        f_fric = -np.outer(cap / denom, np.ones(3)) * self.foot_vel
        f_fric[~active] = 0.0
        self.friction_coef = mu

        # Integrate foot nodes (semi-implicit; foot lives in the plane).
        self.foot_vel = self.foot_vel + (-f_spring + f_fric) * (h / M_FOOT)
        self.foot = self.foot + self.foot_vel * h
        self.foot = self.foot - np.outer((self.foot - o) @ n, n)

        # External + transmitted spring forces on the body.
        self.v = self.v + (m * self.gravity_vec + f_spring.sum(axis=0)) * (h / m)
        torque = np.cross(arms, f_spring).sum(axis=0)
        self.w = self.w + (inertia_inv_w @ torque) * h
        self.spring_force = f_spring

        # ---- normal contact impulses: projected Jacobi (staggered) -------
        cross_an = np.cross(arms, np.broadcast_to(n, (n_c, 3)))
        ann = (
            1.0 / m
            + np.einsum("ij,jk,ik->i", cross_an, inertia_inv_w, cross_an)
        )
        pen = np.maximum(-self.gap, 0.0)
        bias = np.minimum(BAUMGARTE * pen / h, BIAS_CAP)
        touch = self.gap < CONTACT_MARGIN
        lam = np.zeros(n_c)
        for _ in range(SWEEPS):
            u_n = (self.v + np.cross(self.w, arms)) @ n
            delta = np.where(touch, RELAXATION * (bias - u_n) / ann, 0.0)
            lam_new = np.maximum(lam + delta, 0.0)
            delta = lam_new - lam
            lam = lam_new
            impulse = np.outer(delta, n)
            self.v = self.v + impulse.sum(axis=0) * (1.0 / m)
            self.w = self.w + inertia_inv_w @ np.cross(arms, impulse).sum(axis=0)

        self.normal_force = lam / h

        # Integrate the rigid body.
        self.com = self.com + self.v * h
        self.q = _integrate_quaternion(self.q, self.w, h)
        self.active_prev = active
        self._substeps += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time:
            target = self.time
        n = int(round((target - self.time) / SUBSTEP))
        for _ in range(n):
            self._step()
        self.time = target
        return self._frame(target)

    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        rot = _quat_to_matrix(self.q)
        origin = self.com - rot @ self.com_local
        origin_vel = self.v + np.cross(self.w, -rot @ self.com_local)

        active = self.active_prev
        spring_mag = np.linalg.norm(self.spring_force, axis=1)
        value = np.where(active, spring_mag, 0.0)
        target = np.where(active, self.friction_coef * self.normal_force, 0.0)

        fields = {
            "constraint_target": np.asarray(target, dtype=np.float64),
            "constraint_value": np.asarray(value, dtype=np.float64),
            "contact_signed_distance": np.asarray(self.gap, dtype=np.float64),
            "external_force": np.asarray(
                self.mass * self.gravity_vec, dtype=np.float64
            ).reshape(1, 3),
        }
        scalars = {
            "constraint_scale": float(self.mass * GRAVITY),
            "contact_length_scale": float(self.contact_length_scale),
        }
        return StateFrame(
            time=time,
            position=origin.reshape(1, 3),
            velocity=origin_vel.reshape(1, 3),
            orientation=self.q.reshape(1, 4),
            angular_velocity=self.w.reshape(1, 3),
            mass=np.asarray([self.mass]),
            body_id=np.asarray(self.body_id),
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        rot = _quat_to_matrix(self.q)
        origin = self.com - rot @ self.com_local
        return {
            "scene_id": self.scene_id,
            "model": "rigid_body_compliant_feet_stick_slip",
            "substeps": int(self._substeps),
            "substep": SUBSTEP,
            "mu_static": MU_STATIC,
            "mu_kinetic": MU_KINETIC,
            "final_time": float(self.time),
            "final_position": [float(x) for x in origin],
            "final_speed": float(np.linalg.norm(self.v)),
            "max_penetration": float(max(0.0, -self.gap.min())),
        }
