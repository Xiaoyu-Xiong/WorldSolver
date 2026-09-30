"""Generate the deterministic hollow solid rest mesh for the cushion."""
from pathlib import Path

import numpy as np


OUT = Path(__file__).parent
NX, NY, NZ = 8, 6, 5
OUTER = np.array([1.20, 0.80, 0.55], dtype=float)
THICKNESS = 0.09


def grid_vertices():
    axes = [np.linspace(-v / 2.0, v / 2.0, n + 1) for v, n in zip(OUTER, (NX, NY, NZ))]
    vertices = np.array([[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]])
    return vertices


def vid(i, j, k):
    return i + (NX + 1) * (j + (NY + 1) * k)


vertices = grid_vertices()
tets = []
for k in range(NZ):
    for j in range(NY):
        for i in range(NX):
            center = np.array([(i + 0.5) / NX, (j + 0.5) / NY, (k + 0.5) / NZ]) * OUTER - OUTER / 2
            inner_half = OUTER / 2 - THICKNESS
            if np.all(np.abs(center) < inner_half):
                continue
            c = [vid(i, j, k), vid(i + 1, j, k), vid(i, j + 1, k), vid(i + 1, j + 1, k),
                 vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i, j + 1, k + 1), vid(i + 1, j + 1, k + 1)]
            tets.extend([[c[0], c[1], c[3], c[7]], [c[0], c[3], c[2], c[7]],
                         [c[0], c[2], c[6], c[7]], [c[0], c[6], c[4], c[7]],
                         [c[0], c[4], c[5], c[7]], [c[0], c[5], c[1], c[7]]])
tets = np.asarray(tets, dtype=np.int32)
used = np.unique(tets)
remap = np.full(len(vertices), -1, dtype=np.int32)
remap[used] = np.arange(len(used))
vertices = vertices[used]
tets = remap[tets]

# Boundary faces and their orientation-independent occurrence counts.
all_faces = np.vstack((tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [2, 0, 3]]))
keys = np.sort(all_faces, axis=1)
_, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
faces = all_faces[first[counts == 1]]
centers = vertices[faces].mean(axis=1)
outer_half = OUTER / 2
inner = np.max(np.abs(centers) / outer_half, axis=1) < 0.92
inner_faces = faces[inner]
outer_faces = faces[~inner]

vol = np.abs(np.einsum("ij,ij->i", vertices[tets[:, 1]] - vertices[tets[:, 0]],
                       np.cross(vertices[tets[:, 2]] - vertices[tets[:, 0]], vertices[tets[:, 3]] - vertices[tets[:, 0]]))) / 6
nodal_volume = np.zeros(len(vertices))
np.add.at(nodal_volume, tets.ravel(), np.repeat(vol / 4.0, 4))
wall_region = np.zeros(len(vertices), dtype=np.int32)

# `faces` (boundary surface triangles) is required by the platform renderer's
# _load_topology; `surface_faces` is an alias kept for solver-side tooling.
np.savez(OUT / "cushion_solid.npz", vertices=vertices, tetrahedra=tets, faces=faces,
         surface_faces=faces, outer_faces=outer_faces, inner_chamber_faces=inner_faces,
         nodal_volume=nodal_volume, wall_region=wall_region)
with (OUT / "cushion_rest.obj").open("w", encoding="ascii", newline="\n") as f:
    f.write("# Uninflated finite-thickness pneumatic cushion wall\n")
    for v in vertices:
        f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
    for face in faces:
        f.write("f " + " ".join(str(int(q) + 1) for q in face) + "\n")
print(len(vertices), len(tets), len(faces), len(inner_faces))
