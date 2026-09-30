"""Generate the deterministic watertight-ish visual/physical OBJ used by the scaffold."""
from pathlib import Path

verts = []
faces = []

def box(x0, x1, y0, y1, z0, z1):
    n = len(verts)
    verts.extend([(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                  (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)])
    faces.extend(tuple(n+i for i in f) for f in
                 [(1,4,3,2),(5,6,7,8),(1,2,6,5),(2,3,7,6),(3,4,8,7),(4,1,5,8)])

# Continuous thick passive underside: the asymmetric strain-limiting wall.
box(0.0, 0.210, -0.020, 0.020, 0.0, 0.014)
# Six inflated-side chamber lobes separated by narrow compliant hinge gaps.
for i in range(6):
    x0 = 0.008 + i * 0.032
    x1 = x0 + 0.027
    box(x0, x1, -0.024, 0.024, 0.014, 0.050)
# Rounded-looking terminal block and sealed nose.
box(0.200, 0.220, -0.020, 0.020, 0.008, 0.038)

out = Path(__file__).with_name("pneumatic_finger.obj")
lines = ["# Six-chamber asymmetric pneumatic soft finger, meters"]
lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x,y,z in verts]
lines += ["f " + " ".join(map(str, f)) for f in faces]
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
