"""Physically-based rigid body solver for the wrecking ball box stack scene.

Impulse-based rigid body engine:
  - 30 OBB boxes (SAT + reference/incident face clipping manifolds)
  - wrecking ball approximated by a sphere (mesh half width ~0.42 m)
  - 6 chain links joined by ball-socket (point) constraints to a fixed
    anchor under the support beam, forming a pendulum
  - sequential impulses with Coulomb friction, restitution and Baumgarte
    positional correction; static floor plane.

The pendulum starts at the configured bottom-of-arc positions carrying the
rigid-body velocity field of a pendulum released from ~40 degrees on the far
side, so it swings away, returns under gravity and strikes the box stack.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

# ------------------------------------------------------------- vector ops
def vadd(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def vsub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def vmul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def vdot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])
def vlen(a): return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def quat_axes(q):
    """World-space local axes [ax, ay, az] of quaternion (x, y, z, w)."""
    x, y, z, w = q
    ax = (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w))
    ay = (2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w))
    az = (2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))
    return [ax, ay, az]


def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def qnormalize(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def inv3(m):
    """Inverse of a row-major 3x3 matrix given as a 9-tuple."""
    a, b, c, d, e, f, g, h, i = m
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    det = a * A + b * B + c * C
    if abs(det) < 1e-14:
        det = 1e-14 if det >= 0 else -1e-14
    s = 1.0 / det
    return (A * s, (c * h - b * i) * s, (b * f - c * e) * s,
            B * s, (a * i - c * g) * s, (c * d - a * f) * s,
            C * s, (b * g - a * h) * s, (a * e - b * d) * s)


def matvec(m, v):
    return (m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2])


def clip_poly(poly, nrm, off):
    """Keep points with dot(p, nrm) <= off."""
    out = []
    n = len(poly)
    for k in range(n):
        p = poly[k]
        q = poly[(k + 1) % n]
        dp = vdot(p, nrm) - off
        dq = vdot(q, nrm) - off
        if dp <= 0.0:
            out.append(p)
        if dp * dq < 0.0 and abs(dp - dq) > 1e-15:
            t = dp / (dp - dq)
            out.append(vadd(p, vmul(vsub(q, p), t)))
    return out
