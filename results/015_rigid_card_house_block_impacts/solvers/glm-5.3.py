"""Rigid card-house block-impact solver.

Planar (x-z) rigid-body dynamics engine: all frozen geometry is centered on the
y=0 plane and every initial rotation is about the y-axis, so the scene evolves
in a plane while states are exported in full 3D form (quaternion about y).

Physics:
  * semi-implicit Euler velocity integration with gravity and light damping,
  * OBB-vs-OBB separating-axis collision with clipped 2-point contact
    manifolds (Box2D-style reference/incident face clipping),
  * sequential-impulse contact solving with accumulated impulses, Coulomb
    friction (mu = 0.8), restitution (e = 0.1) above a bounce threshold and a
    Baumgarte bias with slop for stable stacking/settling,
  * impact blocks are frozen (static) until their prescribed release time.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Numerical parameters
# ---------------------------------------------------------------------------
SUBSTEPS_PER_SAMPLE = 20          # dt = 0.02 / 20 = 1 ms
SOLVER_ITERATIONS = 24
BAUMGARTE = 0.20                  # positional bias factor
PENETRATION_SLOP = 5.0e-4         # m
BOUNCE_THRESHOLD = 0.5            # m/s; below this contacts are inelastic
NEAR_CONTACT_MARGIN = 0.05        # m; gaps up to this are reported
LINEAR_DAMPING = 0.02             # 1/s
ANGULAR_DAMPING = 0.05            # 1/s
CONTACT_LENGTH_SCALE = 0.03       # m, card thickness


def _rot(angle: float):
    c = math.cos(angle)
    s = math.sin(angle)
    return c, s


class Body:
    """Planar rigid body: center (x, z), CCW angle alpha in the (x, z) plane.

    alpha is the standard 2D counter-clockwise angle expressed in coordinates
    (x, z); it maps to a rotation about the world +y axis by theta = -alpha.
    Local box axes: e1 = local +x, e2 = local +z.
    """

    __slots__ = (
        "index", "half", "mass", "inv_mass", "inertia", "inv_inertia",
        "cx", "cz", "alpha", "vx", "vz", "omega", "static", "release_time",
        "is_impact_block", "mass0",
    )

    def __init__(self, index, half, mass, position, alpha, static,
                 release_time=0.0, is_impact_block=False):
        self.index = index
        self.half = (float(half[0]), float(half[1]))
        self.mass0 = float(mass)
        self.mass = float(mass)
        self.release_time = float(release_time)
        self.is_impact_block = bool(is_impact_block)
        self.cx = float(position[0])
        self.cz = float(position[1])
        self.alpha = float(alpha)
        self.vx = 0.0
        self.vz = 0.0
        self.omega = 0.0
        self.static = bool(static)
        ha, hc = self.half
        # inertia about y: m * ((2 ha)^2 + (2 hc)^2) / 12
        self.inertia = mass * ((2.0 * ha) ** 2 + (2.0 * hc) ** 2) / 12.0
        self._refresh_inverse()

    def _refresh_inverse(self):
        if self.static or self.mass <= 0.0:
            self.inv_mass = 0.0
            self.inv_inertia = 0.0
        else:
            self.inv_mass = 1.0 / self.mass
            self.inv_inertia = 1.0 / self.inertia

    def set_static(self, static: bool):
        self.static = static
        if static:
            self.vx = 0.0
            self.vz = 0.0
            self.omega = 0.0
        self._refresh_inverse()

    def axes(self):
        c, s = _rot(self.alpha)
        # e1 = local +x in (x, z) coords, e2 = local +z
        return (c, s), (-s, c)

    def aabb(self, margin: float):
        (e1x, e1z), (e2x, e2z) = self.axes()
        ha, hc = self.half
        rx = abs(e1x) * ha + abs(e2x) * hc + margin
        rz = abs(e1z) * ha + abs(e2z) * hc + margin
        return (self.cx - rx, self.cz - rz, self.cx + rx, self.cz + rz)

    def point_velocity(self, px: float, pz: float):
        rx = px - self.cx
        rz = pz - self.cz
        # CCW rotation in (x, z): v_point = v + omega * perp(r), perp = (-rz, rx)
        return (self.vx - self.omega * rz, self.vz + self.omega * rx)


class Contact:
    __slots__ = ("a", "b", "px", "pz", "nx", "nz", "pen",
                 "mass_n", "mass_t", "bias", "pn", "pt", "rn_a", "rn_b",
                 "rt_a", "rt_b", "vx0")

    def __init__(self, body_a, body_b, px, pz, nx, nz, pen):
        self.a = body_a
        self.b = body_b
        self.px = px
        self.pz = pz
        self.nx = nx
        self.nz = nz
        self.pen = pen
        self.pn = 0.0
        self.pt = 0.0

    def prepare(self, dt: float, mu: float, restitution: float):
        a, b = self.a, self.b
        rax = self.px - a.cx
        raz = self.pz - a.cz
        rbx = self.px - b.cx
        rbz = self.pz - b.cz
        # rotational velocity coefficients (perp(r) . n and perp(r) . t)
        self.rn_a = -raz * self.nx + rax * self.nz
        self.rn_b = -rbz * self.nx + rbx * self.nz
        tx, tz = -self.nz, self.nx
        self.rt_a = -raz * tx + rax * tz
        self.rt_b = -rbz * tx + rbx * tz
        inv_a, inv_b = a.inv_mass, b.inv_mass
        ii_a, ii_b = a.inv_inertia, b.inv_inertia
        self.mass_n = 1.0 / (inv_a + inv_b + self.rn_a * self.rn_a * ii_a
                             + self.rn_b * self.rn_b * ii_b)
        self.mass_t = 1.0 / (inv_a + inv_b + self.rt_a * self.rt_a * ii_a
                             + self.rt_b * self.rt_b * ii_b)
        # relative normal velocity (b relative to a along n from a to b)
        vax, vaz = a.point_velocity(self.px, self.pz)
        vbx, vbz = b.point_velocity(self.px, self.pz)
        self.vx0 = (vbx - vax) * self.nx + (vbz - vaz) * self.nz
        bias_speed = (BAUMGARTE / dt) * max(self.pen - PENETRATION_SLOP, 0.0)
        if restitution > 0.0 and self.vx0 < -BOUNCE_THRESHOLD:
            bounce = -restitution * self.vx0
            if bounce > bias_speed:
                bias_speed = bounce
        self.bias = bias_speed

    def relative_velocity(self):
        vax, vaz = self.a.point_velocity(self.px, self.pz)
        vbx, vbz = self.b.point_velocity(self.px, self.pz)
        return vbx - vax, vbz - vaz

    def apply_impulse(self, dpx: float, dpz: float):
        a, b = self.a, self.b
        rax = self.px - a.cx
        raz = self.pz - a.cz
        rbx = self.px - b.cx
        rbz = self.pz - b.cz
        if a.inv_mass > 0.0:
            a.vx -= dpx * a.inv_mass
            a.vz -= dpz * a.inv_mass
            a.omega -= (rax * dpz - raz * dpx) * a.inv_inertia
        if b.inv_mass > 0.0:
            b.vx += dpx * b.inv_mass
            b.vz += dpz * b.inv_mass
            b.omega += (rbx * dpz - rbz * dpx) * b.inv_inertia


# ---------------------------------------------------------------------------
# Collision detection (SAT + reference/incident face clipping)
# ---------------------------------------------------------------------------
def _box_support_axes(body: Body):
    (e1x, e1z), (e2x, e2z) = body.axes()
    return (e1x, e1z), (e2x, e2z)


def _axis_separation(a: Body, b: Body, ax: float, az: float):
    """Separation of boxes a and b along unit axis (ax, az); >0 means apart."""
    dx = b.cx - a.cx
    dz = b.cz - a.cz
    dist = dx * ax + dz * az
    (a1x, a1z), (a2x, a2z) = _box_support_axes(a)
    (b1x, b1z), (b2x, b2z) = _box_support_axes(b)
    ha_a, hc_a = a.half
    ha_b, hc_b = b.half
    proj_a = abs(a1x * ax + a1z * az) * ha_a + abs(a2x * ax + a2z * az) * hc_a
    proj_b = abs(b1x * ax + b1z * az) * ha_b + abs(b2x * ax + b2z * az) * hc_b
    return abs(dist) - proj_a - proj_b, dist


def collide(a: Body, b: Body, slop: float):
    """Return list of Contact(a, b, ...) with normal from a to b, or [].

    Also returns the minimum signed separation even when no contact exists
    (used for near-contact reporting).
    """
    axes_a = _box_support_axes(a)
    axes_b = _box_support_axes(b)
    best_sep = -float("inf")
    best_axis = None
    best_from_a = True
    for (ax, az), from_a in (
        (axes_a[0], True), (axes_a[1], True),
        (axes_b[0], False), (axes_b[1], False),
    ):
        sep, dist = _axis_separation(a, b, ax, az)
        if sep > best_sep:
            best_sep = sep
            best_axis = (ax, az)
            best_from_a = from_a
    if best_sep > NEAR_CONTACT_MARGIN:
        return [], best_sep
    if best_sep > 0.0:
        return [], best_sep  # near but not touching

    ax, az = best_axis
    dx = b.cx - a.cx
    dz = b.cz - a.cz
    dist = dx * ax + dz * az
    nx, nz = (ax, az) if dist >= 0.0 else (-ax, -az)

    ref = a if best_from_a else b
    inc = b if best_from_a else a
    # normal pointing from the reference box toward the incident box
    if best_from_a:
        nrx, nrz = nx, nz
    else:
        nrx, nrz = -nx, -nz
    # reference face center: extreme point of ref toward the incident box
    (r1x, r1z), (r2x, r2z) = _box_support_axes(ref)
    ha_r, hc_r = ref.half
    proj = r1x * nrx + r1z * nrz
    # nr is parallel to one of ref's axes; pick that axis's half extent
    along_e1 = abs(proj) > 0.5
    if along_e1:
        half_normal = ha_r
        tangent = (r2x, r2z)
        half_tangent = hc_r
    else:
        half_normal = hc_r
        tangent = (r1x, r1z)
        half_tangent = ha_r
    fcx = ref.cx + nrx * half_normal
    fcz = ref.cz + nrz * half_normal

    # incident face: face of inc most anti-parallel to nr
    (i1x, i1z), (i2x, i2z) = _box_support_axes(inc)
    ha_i, hc_i = inc.half
    d1 = i1x * nrx + i1z * nrz
    d2 = i2x * nrx + i2z * nrz
    if d1 <= d2:
        inc_normal = (i1x, i1z)
        inc_half = ha_i
        inc_tan = (i2x, i2z)
        inc_half_t = hc_i
    else:
        inc_normal = (i2x, i2z)
        inc_half = hc_i
        inc_tan = (i1x, i1z)
        inc_half_t = ha_i
    sign = 1.0 if (inc_normal[0] * nrx + inc_normal[1] * nrz) <= 0.0 else -1.0
    ifcx = inc.cx - sign * inc_normal[0] * inc_half
    ifcz = inc.cz - sign * inc_normal[1] * inc_half
    p1 = (ifcx - inc_tan[0] * inc_half_t, ifcz - inc_tan[1] * inc_half_t)
    p2 = (ifcx + inc_tan[0] * inc_half_t, ifcz + inc_tan[1] * inc_half_t)

    # clip segment p1-p2 against reference face side planes
    tx, tz = tangent
    side1 = fcx * tx + fcz * tz + half_tangent  # keep dot(p, t) <= side1
    side2 = fcx * tx + fcz * tz - half_tangent  # keep dot(p, t) >= side2

    def clip_le(point, limit):
        return point[0] * tx + point[1] * tz - limit

    d1v = clip_le(p1, side1)
    d2v = clip_le(p2, side1)
    if d1v <= 0.0 and d2v <= 0.0:
        pass
    elif d1v > 0.0 and d2v > 0.0:
        return [], best_sep
    elif d1v > 0.0:
        t = d1v / (d1v - d2v)
        p1 = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    else:
        t = d2v / (d2v - d1v)
        p2 = (p2[0] + t * (p1[0] - p2[0]), p2[1] + t * (p1[1] - p2[1]))

    d1v = -clip_le(p1, side2)  # dot(p, t) - side2 >= 0
    d2v = -clip_le(p2, side2)
    if d1v <= 0.0 and d2v <= 0.0:
        pass
    elif d1v > 0.0 and d2v > 0.0:
        return [], best_sep
    elif d1v > 0.0:
        t = d1v / (d1v - d2v)
        p1 = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
    else:
        t = d2v / (d2v - d1v)
        p2 = (p2[0] + t * (p1[0] - p2[0]), p2[1] + t * (p1[1] - p2[1]))

    contacts = []
    for (px, pz) in (p1, p2):
        s = (px - fcx) * nrx + (pz - fcz) * nrz
        if s <= slop:
            contacts.append(Contact(a, b, px, pz, nx, nz, -s))
    return contacts, best_sep


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class BenchmarkSolver:
    def __init__(self):
        self.bodies = []
        self.floor = None
        self.gravity = (0.0, -9.81)
        self.mu = 0.8
        self.restitution = 0.1
        self.time = 0.0
        self.dt_sub = 0.001
        self._last_contacts = []
        self._last_impulses = []
        self._diagnostics = {}
        self._initial_frame = None

    # -- setup --------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self._initial_frame = frame
        meta = scene.metadata()
        g = meta.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = (float(g[0]), float(g[2]))
        self.mu = float(meta.get("friction_coefficient", 0.8))
        self.restitution = float(meta.get("restitution", 0.1))

        positions = np.asarray(frame.position, dtype=float)
        orientations = np.asarray(frame.orientation, dtype=float)
        masses = np.asarray(frame.mass, dtype=float)
        release = np.asarray(frame.attributes["release_time"], dtype=float)
        is_block = np.asarray(frame.attributes["is_impact_block"], dtype=float)

        half_extents = self._geometry_half_extents(scene)
        n = len(positions)
        self.bodies = []
        for i in range(n):
            q = orientations[i]
            # quaternion (0, qy, 0, qw) -> rotation about y by theta
            qy = q[1]
            qw = q[3]
            norm = math.sqrt(qy * qy + qw * qw)
            qy /= norm
            qw /= norm
            theta = 2.0 * math.atan2(qy, qw)
            alpha = -theta  # CCW angle in (x, z) coords
            half = half_extents.get(i, (0.1, 0.1))
            static = bool(is_block[i] > 0.5 and release[i] > 0.0)
            self.bodies.append(
                Body(i, half, float(masses[i]), (positions[i, 0], positions[i, 2]),
                     alpha, static, float(release[i]), bool(is_block[i] > 0.5))
            )
        # static floor boundary
        floor_half, floor_pos = self._floor_geometry(scene)
        self.floor = Body(-1, floor_half, 0.0, floor_pos, 0.0, True)

        sample_dt = 0.02
        try:
            sample_dt = float(scene.config["sample_dt"])
        except (KeyError, TypeError, ValueError):
            pass
        self.dt_sub = sample_dt / SUBSTEPS_PER_SAMPLE
        self.time = 0.0
        self._last_contacts = []
        self._last_impulses = []
        self._diagnostics = {
            "method": "sequential-impulse planar rigid body dynamics",
            "bodies": n + 1,
            "substep_dt": self.dt_sub,
        }

    def _geometry_half_extents(self, scene):
        """half (x, z) per body_id from the frozen scene geometry."""
        result = {}
        try:
            objects = scene.config["scene_geometry"]["objects"]
        except (KeyError, TypeError):
            objects = None
        if objects:
            for obj in objects:
                binding = obj.get("state_binding") or {}
                if binding.get("selector") != "body_id":
                    continue
                geom = obj.get("geometry") or {}
                he = geom.get("half_extents")
                if he and len(he) == 3:
                    result[int(binding["value"])] = (float(he[0]), float(he[2]))
        if len(result) < len(np.asarray(self._initial_frame.position)):
            defaults = {i: (0.45, 0.015) for i in range(10)}
            defaults.update({10: (0.16, 0.16), 11: (0.16, 0.16), 12: (0.19, 0.19)})
            for k, v in defaults.items():
                result.setdefault(k, v)
        return result

    def _floor_geometry(self, scene):
        """Half extents (x, z) and center (x, z) of the static floor box."""
        try:
            objects = scene.config["scene_geometry"]["objects"]
            for obj in objects:
                if obj.get("id") == "floor":
                    he = obj["geometry"]["half_extents"]
                    tr = obj.get("transform", {}).get("translation", [0.0, 0.0, -0.05])
                    return (float(he[0]), float(he[2])), (float(tr[0]), float(tr[2]))
        except (KeyError, TypeError, ValueError):
            pass
        return (3.2, 0.05), (0.0, -0.05)

    # -- stepping -----------------------------------------------------------
    def _all_bodies(self):
        return [self.floor] + self.bodies if self.floor is not None else list(self.bodies)

    def _release_due(self, t_now: float):
        for body in self.bodies:
            if body.is_impact_block and body.static and t_now >= body.release_time - 1e-12:
                body.set_static(False)

    def _step(self, dt: float):
        self._release_due(self.time)
        bodies = self._all_bodies()

        # integrate forces (gravity + damping) for dynamic bodies
        lin_scale = math.exp(-LINEAR_DAMPING * dt)
        ang_scale = math.exp(-ANGULAR_DAMPING * dt)
        gx, gz = self.gravity
        for b in bodies:
            if b.inv_mass <= 0.0:
                continue
            b.vx += gx * dt
            b.vz += gz * dt
            b.vx *= lin_scale
            b.vz *= lin_scale
            b.omega *= ang_scale

        # collision detection
        contacts = []
        n = len(bodies)
        for i in range(n):
            a = bodies[i]
            ax0, az0, ax1, az1 = a.aabb(0.0)
            for j in range(i + 1, n):
                b = bodies[j]
                if a.inv_mass <= 0.0 and b.inv_mass <= 0.0:
                    continue
                bx0, bz0, bx1, bz1 = b.aabb(0.0)
                if ax0 > bx1 or bx0 > ax1 or az0 > bz1 or bz0 > az1:
                    continue
                found, _sep = collide(a, b, PENETRATION_SLOP)
                contacts.extend(found)

        for c in contacts:
            c.prepare(dt, self.mu, self.restitution)

        # sequential impulses
        for _ in range(SOLVER_ITERATIONS):
            for c in contacts:
                if c.a.inv_mass <= 0.0 and c.b.inv_mass <= 0.0:
                    continue
                # normal
                rvx, rvz = c.relative_velocity()
                vn = rvx * c.nx + rvz * c.nz
                dλ = c.mass_n * (c.bias - vn)
                new_pn = max(c.pn + dλ, 0.0)
                dλ = new_pn - c.pn
                c.pn = new_pn
                if dλ != 0.0:
                    c.apply_impulse(dλ * c.nx, dλ * c.nz)
                # friction
                tx, tz = -c.nz, c.nx
                rvx, rvz = c.relative_velocity()
                vt = rvx * tx + rvz * tz
                dλt = c.mass_t * (-vt)
                max_f = self.mu * c.pn
                new_pt = min(max(c.pt + dλt, -max_f), max_f)
                dλt = new_pt - c.pt
                c.pt = new_pt
                if dλt != 0.0:
                    c.apply_impulse(dλt * tx, dλt * tz)

        # record contact impulses for reporting (instantaneous force = P/dt)
        self._last_contacts = contacts
        self._last_impulses = []
        for c in contacts:
            fx = c.pn * c.nx - c.pt * c.nz
            fz = c.pn * c.nz + c.pt * c.nx
            self._last_impulses.append((fx / dt, fz / dt))

        # integrate positions (with finite-state safety clamps)
        for b in bodies:
            if b.inv_mass <= 0.0:
                continue
            b.cx += b.vx * dt
            b.cz += b.vz * dt
            b.alpha += b.omega * dt
            speed2 = b.vx * b.vx + b.vz * b.vz
            if speed2 > 2500.0:  # 50 m/s cap
                scale = 50.0 / math.sqrt(speed2)
                b.vx *= scale
                b.vz *= scale
            if b.omega > 200.0:
                b.omega = 200.0
            elif b.omega < -200.0:
                b.omega = -200.0
        self.time += dt

    # -- sampling -----------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1e-9:
            dt = min(self.dt_sub, target - self.time)
            self._step(dt)
            guard += 1
            if guard > 10 * SUBSTEPS_PER_SAMPLE + 10:
                self.time = target
                break
        return self._build_frame(self.time)

    def _query_contact_gaps(self):
        """Signed gaps at active and near-contact samples at the current state."""
        bodies = self._all_bodies()
        gaps = []
        n = len(bodies)
        for i in range(n):
            a = bodies[i]
            ax0, az0, ax1, az1 = a.aabb(NEAR_CONTACT_MARGIN)
            for j in range(i + 1, n):
                b = bodies[j]
                bx0, bz0, bx1, bz1 = b.aabb(NEAR_CONTACT_MARGIN)
                if ax0 > bx1 or bx0 > ax1 or az0 > bz1 or bz0 > az1:
                    continue
                contacts, sep = collide(a, b, PENETRATION_SLOP)
                if contacts:
                    for c in contacts:
                        gaps.append(-c.pen)
                elif sep < NEAR_CONTACT_MARGIN:
                    gaps.append(sep)
        return gaps

    def _build_frame(self, t: float):
        n = len(self.bodies)
        position = np.zeros((n, 3), dtype=float)
        velocity = np.zeros((n, 3), dtype=float)
        orientation = np.zeros((n, 4), dtype=float)
        angular_velocity = np.zeros((n, 3), dtype=float)
        mass = np.zeros(n, dtype=float)
        body_id = np.zeros(n, dtype=int)
        for i, b in enumerate(self.bodies):
            position[i, 0] = b.cx
            position[i, 2] = b.cz
            velocity[i, 0] = b.vx
            velocity[i, 2] = b.vz
            theta = -b.alpha
            orientation[i, 1] = math.sin(0.5 * theta)
            orientation[i, 3] = math.cos(0.5 * theta)
            angular_velocity[i, 1] = -b.omega
            mass[i] = b.mass0
            body_id[i] = b.index

        gx, gz = self.gravity
        external_force = np.zeros((n, 3), dtype=float)
        external_force[:, 2] = mass * gz
        external_force[:, 0] = mass * gx

        gaps = self._query_contact_gaps()
        if gaps:
            contact_signed_distance = np.asarray(gaps, dtype=float)
        else:
            contact_signed_distance = np.zeros(0, dtype=float)

        if self._last_impulses:
            interaction_force = np.zeros((len(self._last_impulses), 3), dtype=float)
            for k, (fx, fz) in enumerate(self._last_impulses):
                interaction_force[k, 0] = fx
                interaction_force[k, 2] = fz
        else:
            interaction_force = np.zeros((0, 3), dtype=float)

        attributes = {}
        if self._initial_frame is not None:
            for name in ("release_time", "is_impact_block"):
                if name in self._initial_frame.attributes:
                    attributes[name] = np.asarray(
                        self._initial_frame.attributes[name], dtype=float
                    )

        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields={
                "contact_signed_distance": contact_signed_distance,
                "external_force": external_force,
                "interaction_force": interaction_force,
            },
            scalars={
                "contact_length_scale": CONTACT_LENGTH_SCALE,
                "friction_coefficient": self.mu,
                "restitution": self.restitution,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return dict(self._diagnostics)
