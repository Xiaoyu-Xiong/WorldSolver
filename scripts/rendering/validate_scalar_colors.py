"""Small GPU regression for scalar-colored meshes; no physics or model calls.

Run this module in the independent Luisa environment under a memory-capped
service. All generated test geometry, pictures, and the receipt go to --output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import trimesh

from worldsolver.config import load_yaml
from worldsolver.contracts import StateFrame
from worldsolver.contracts.scene_geometry import SceneGeometry, load_scene_geometry

from .luisa_renderer import LuisaTrajectoryRenderer, _scalar_attribute_texture
from .render_spec import RenderSpec, load_render_spec


def _frame(vertices, values, time):
    return StateFrame(
        time=time,
        position=vertices,
        velocity=np.zeros_like(vertices),
        body_id=np.ones(len(vertices), dtype=int),
        attributes={"saturation": values},
    )


def _patch(image, x, y):
    # Synthetic orthographic camera: world height 1.2, width 2.4.
    px = round((x / 2.4 + 0.5) * image.shape[1])
    py = round((0.5 - y / 1.2) * image.shape[0])
    return np.median(image[py - 3 : py + 4, px - 3 : px + 4], axis=(0, 1))


def synthetic_test(output):
    output.mkdir(parents=True, exist_ok=True)
    columns = np.asarray([-1.05, -0.5, 0.0, 0.5, 1.05])
    vertices = np.asarray([[x, y, 0.0] for y in [0.05, 0.45] for x in columns])
    faces = np.asarray([[i, i + 1, i + 6] for i in range(4)] + [[i, i + 6, i + 5] for i in range(4)])
    np.savez(output / "field.npz", vertices=vertices, faces=faces)
    low = np.asarray([0.12, 0.30, 0.80])
    high = np.asarray([0.82, 0.12, 0.08])
    endpoints = np.rint(np.asarray([low, high]) * 255) / 255.0
    objects = [
        {
            "id": "field",
            "semantic": "deformable",
            "motion": "dynamic",
            "geometry": {"type": "triangle_mesh", "topology_asset": str(output / "field.npz")},
            "state_binding": {"selector": "body_id", "value": 1},
        }
    ]
    bindings = [
        {
            "object_id": "field",
            "method": "deformed_mesh",
            "scalar_attribute": "saturation",
            "scalar_range": [0.0, 1.0],
            "scalar_color_low": low.tolist(),
            "scalar_color_high": high.tolist(),
        }
    ]
    overrides = {"field": {"roughness": 1.0}}
    for index, (x, fraction) in enumerate(zip([-0.8, 0.0, 0.8], [0.0, 0.5, 1.0], strict=True)):
        name = f"reference_{index}"
        patch = trimesh.Trimesh(
            vertices=[[x - 0.20, -0.45, 0], [x + 0.20, -0.45, 0], [x + 0.20, -0.1, 0], [x - 0.20, -0.1, 0]],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )
        patch.export(output / f"{name}.obj")
        objects.append(
            {
                "id": name,
                "semantic": "boundary",
                "motion": "static",
                "geometry": {"type": "triangle_mesh", "asset": str(output / f"{name}.obj")},
            }
        )
        bindings.append({"object_id": name, "method": "rigid_mesh"})
        overrides[name] = {
            "color_rgb": ((1 - fraction) * endpoints[0] + fraction * endpoints[1]).tolist(),
            "roughness": 1.0,
        }
    geometry = SceneGeometry.model_validate({"objects": objects})
    spec = RenderSpec.model_validate({"profile": "multiphysics", "bindings": bindings})
    config = {
        "camera": {
            "position": [0, 0, 3],
            "look_at": [0, 0, 0],
            "up": [0, 1, 0],
            "projection": "orthographic",
            "orthographic_height": 1.2,
            "width": 640,
            "height": 320,
        },
        "renderer": {"spp": 256, "denoise": False},
        "luisa_environment_color": [0.5, 0.5, 0.5],
        "luisa_lights": [],
        "render_materials": overrides,
    }
    clipped = np.tile([-1, 0, 0.5, 1, 2], 2)
    test_frame = _frame(vertices, clipped, 0)
    uvs, texture = _scalar_attribute_texture(test_frame, geometry.objects[0], spec.bindings[0], len(vertices))
    np.testing.assert_allclose(uvs[:, 0], 0.25 + 0.5 * np.clip(clipped, 0, 1))
    np.testing.assert_allclose(uvs[:, 1], 0.5)
    np.testing.assert_array_equal(texture[0], np.rint(np.asarray([low, high]) * 255))

    class MatchedReferenceRenderer(LuisaTrajectoryRenderer):
        def _surface(self, primitive, rgba=None, object_id=None, scalar_texture=None):
            # pyrender assigns vertex-colored meshes a different default
            # metallic factor from ordinary meshes. Reference patches must
            # have the same BSDF to isolate texture/color interpolation.
            if object_id == "field":
                self.reference_metallic = primitive.material.metallicFactor
            else:
                primitive.material.metallicFactor = self.reference_metallic
            return super()._surface(primitive, rgba, object_id, scalar_texture=scalar_texture)

    renderer = MatchedReferenceRenderer(config, spec, geometry, output / "synthetic.mp4")
    images = []
    cache_sizes = []
    try:
        for index, values in enumerate([np.zeros(10), clipped, 1 - clipped, np.full(10, 0.5)]):
            image = renderer.render(_frame(vertices, values, float(index))).copy()
            imageio.imwrite(output / f"frame_{index}.png", image)
            images.append(image)
            cache_sizes.append(len(renderer.surfaces))
    finally:
        renderer.close()
    checks = []
    # Independent constant linear-RGB materials below the gradient provide a
    # GPU reference: this catches an accidental sRGB decoding of diagnostic RGB.
    for frame_index, expected in [(0, [0, 0, 0]), (1, [0, 1, 2]), (2, [2, 1, 0]), (3, [1, 1, 1])]:
        for x, reference in zip([-0.8, 0.0, 0.8], expected, strict=True):
            actual = _patch(images[frame_index], x, 0.25)
            target = _patch(images[frame_index], [-0.8, 0.0, 0.8][reference], -0.275)
            error = float(np.max(np.abs(actual - target)))
            checks.append(
                {
                    "frame": frame_index,
                    "world_x": x,
                    "rgb": actual.tolist(),
                    "reference_rgb": target.tolist(),
                    "max_uint8_error": error,
                }
            )
    assert max(check["max_uint8_error"] for check in checks) <= 8, checks
    assert len(set(cache_sizes)) == 1, cache_sizes
    return {
        "passed": True,
        "pixel_tolerance_uint8": 8,
        "checks": checks,
        "material_cache_sizes": cache_sizes,
        "frames": 4,
    }


def official_mesh_test(output, task):
    scaffold = Path(__file__).resolve().parents[2] / "scaffolds" / task
    geometry = load_scene_geometry(scaffold / "scene_geometry.yaml")
    spec = load_render_spec(scaffold / "render_spec.yaml")
    item = next(item for item in geometry.objects if item.id == "sponge")
    binding = next(binding for binding in spec.bindings if binding.object_id == "sponge")
    geometry = geometry.model_copy(update={"objects": [item]})
    spec = spec.model_copy(update={"bindings": [binding]})
    mesh = trimesh.load_mesh(item.geometry.topology_asset or item.geometry.asset, process=False)
    vertices = np.asarray(mesh.vertices)
    scalar = (vertices[:, 0] - vertices[:, 0].min()) / np.ptp(vertices[:, 0])
    config = load_yaml(scaffold / "config.yaml")
    config["renderer"] = {"spp": 32}
    output.mkdir(parents=True, exist_ok=True)
    renderer = LuisaTrajectoryRenderer(config, spec, geometry, output / "sponge_only.mp4")
    hashes = []
    try:
        for index, values in enumerate([scalar, 1 - scalar]):
            image = renderer.render(_frame(vertices, values, float(index))).copy()
            imageio.imwrite(output / f"frame_{index}.png", image)
            hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
    finally:
        renderer.close()
    assert hashes[0] != hashes[1]
    return {
        "passed": True,
        "task": task,
        "scope": "original sponge mesh and camera only, synthetic scalar states; no simulation",
        "vertices": len(vertices),
        "faces": len(mesh.faces),
        "frames": 2,
        "image_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "render-only scalar color regression; no benchmark scoring, simulation, or model calls",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": hashlib.sha256(Path(__file__).with_name("luisa_renderer.py").read_bytes()).hexdigest(),
        "synthetic": synthetic_test(args.output / "synthetic"),
        "official_meshes": [
            official_mesh_test(args.output / task, task)
            for task in ["119_porous_sponge_absorption_swelling", "121_sponge_saturation_and_wringing"]
        ],
    }
    (args.output / "regression.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
