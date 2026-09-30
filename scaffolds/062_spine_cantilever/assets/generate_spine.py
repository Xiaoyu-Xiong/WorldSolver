"""Generate the deterministic reference surface for the segmented spine arm."""
from pathlib import Path

out = Path(__file__).with_name("spine_arm.obj")
vertices = []
faces = []


def box(x0, x1, hy, hz):
    base = len(vertices) + 1
    vertices.extend([
        (x0, -hy, -hz), (x1, -hy, -hz), (x1, hy, -hz), (x0, hy, -hz),
        (x0, -hy, hz), (x1, -hy, hz), (x1, hy, hz), (x0, hy, hz),
    ])
    faces.extend(tuple(base + i for i in f) for f in [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ])


# Broad anchored root followed by ten tapering vertebrae and narrow flexible necks.
box(0.0, 0.13, 0.075, 0.115)
x = 0.13
for i in range(10):
    neck = 0.032
    body = 0.092 - 0.003 * i
    half_y = 0.075 - 0.0025 * i
    half_z = 0.105 - 0.0065 * i
    box(x - 0.004, x + neck + 0.004, half_y * 0.58, 0.027)
    x += neck
    box(x - 0.004, x + body + 0.004, half_y, half_z)
    x += body

lines = ["# segmented volumetric rubber spine surface"]
lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in vertices]
lines += ["f " + " ".join(map(str, f)) for f in faces]
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
