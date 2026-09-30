"""Generate the deterministic connected Octocat surface used by this scaffold."""
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes
import trimesh


def segment_distance(points, a, b):
    ab = b - a
    t = np.clip(np.sum((points - a) * ab, axis=-1) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * ab), axis=-1)


def main():
    lo = np.array([-1.15, -1.15, -0.82])
    hi = np.array([1.15, 1.15, 1.28])
    shape = (104, 104, 96)
    axes = [np.linspace(lo[i], hi[i], shape[i]) for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    p = np.stack((x, y, z), axis=-1)
    field = np.full(shape, 10.0)

    # Round head/body and two pointed cat ears.
    q = (p - np.array([0.0, 0.0, 0.35])) / np.array([0.55, 0.50, 0.68])
    field = np.minimum(field, np.linalg.norm(q, axis=-1) - 1.0)
    for cx in (-0.34, 0.34):
        tip = np.array([cx, 0.0, 1.18])
        base = np.array([cx * 0.72, 0.0, 0.78])
        d = segment_distance(p, base, tip) / 0.18
        field = np.minimum(field, d - 1.0)

    # Eight distinct, outward-curving tentacles in the horizontal plane.
    for i in range(8):
        a = i * np.pi / 4.0 + np.pi / 8.0
        radial = np.array([np.cos(a), np.sin(a), 0.0])
        tangent = np.array([-np.sin(a), np.cos(a), 0.0])
        pts = [
            np.array([0.32 * radial[0], 0.32 * radial[1], -0.05]),
            np.array([0.62 * radial[0], 0.62 * radial[1], -0.30]),
            np.array([0.86 * radial[0], 0.86 * radial[1], -0.42]) + 0.12 * tangent,
            np.array([1.02 * radial[0], 1.02 * radial[1], -0.28]) + 0.28 * tangent,
        ]
        for j in range(3):
            radius = 0.15 - 0.025 * j
            field = np.minimum(field, segment_distance(p, pts[j], pts[j + 1]) / radius - 1.0)

    spacing = (hi - lo) / (np.array(shape) - 1)
    vertices, faces, _, _ = marching_cubes(field, 0.0, spacing=spacing)
    vertices += lo
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.export(Path(__file__).with_name("octocat.obj"))


if __name__ == "__main__":
    main()
