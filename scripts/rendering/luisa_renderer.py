"""Trajectory-only LuisaRender backend, independent of the Genesis simulator.

The authoritative WorldSolver mesh builders still determine geometry, transforms,
colors, and camera framing.  Only the drawing/lighting backend changes.  Luisa's
compiled binding is loaded from a separate runtime; its caches live beside the
new video and never in the original Genesis installation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np

from worldsolver.trajectory import TrajectoryReader


class _MeshCollector:
    def __init__(self):
        self.meshes = []
        self.object_id = None

    def add(self, mesh, pose=None):
        self.meshes.append((mesh, np.eye(4) if pose is None else np.asarray(pose), self.object_id))


def _load_binding(destination: Path):
    default = Path(__file__).resolve().parents[3] / "worldsolver_runtime/luisa_runtime/bin"
    binary_dir = Path(os.environ.get("LUISA_RENDER_BIN", str(default))).resolve()
    if not list(binary_dir.glob("LuisaRenderPy*.so")):
        raise RuntimeError(
            f"LuisaRenderPy is unavailable in {binary_dir}. Set LUISA_RENDER_BIN to an "
            "independent CPython-compatible runtime, or select --renderer pyrender."
        )
    # Native Luisa stores ctx_* under context_path and loads plugins there. A
    # private directory of read-only binary symlinks keeps caches out of Genesis.
    context_dir = destination.parent / ".luisa_runtime"
    context_dir.mkdir(parents=True, exist_ok=True)
    for source in binary_dir.iterdir():
        if source.is_file():
            target = context_dir / source.name
            # An earlier sandbox may have used /opt/luisa/bin; allow a later
            # cache-first host rerender to repair only this private symlink.
            if target.is_symlink() and target.readlink() != source:
                target.unlink()
            if not target.exists():
                target.symlink_to(source)
    sys.path.insert(0, str(binary_dir))
    binding = importlib.import_module("LuisaRenderPy")
    return binding, context_dir


def _linear_rgb(rgb):
    rgb = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _scalar_attribute_texture(frame, item, binding, vertex_count):
    """Encode a declared two-color scalar map without averaging vertex values.

    Luisa's mesh binding has UVs but no vertex-color channel. For a linear
    low/high map, barycentric interpolation of these UVs followed by bilinear
    texture sampling is the same linear RGB map. Both endpoint texels use the
    legacy renderer's uint8 color precision (at most 0.5/255 rounding error).
    """
    from .renderer import _binding_indices

    indices = _binding_indices(frame, item)
    values = np.asarray(frame.attributes[binding.scalar_attribute], dtype=float)[indices]
    if values.shape != (vertex_count,) or not np.isfinite(values).all():
        raise ValueError(f"Scalar color/vertex mismatch or non-finite values for {item.id}")
    low, high = binding.scalar_range
    fraction = np.clip((values - low) / (high - low), 0.0, 1.0)
    # A 2x1 image's texel centers are u=1/4 and u=3/4. Staying between
    # them also prevents repeat-addressing from blending the opposite edge.
    uvs = np.column_stack((0.25 + 0.5 * fraction, np.full(vertex_count, 0.5)))
    texture = np.rint(np.asarray([[binding.scalar_color_low, binding.scalar_color_high]]) * 255).astype(np.uint8)
    return uvs, texture


def _cylinder_surface_attributes(vertices, faces, uvs):
    """Separate planar caps from the radial wall without moving triangles.

    Shared rim vertices otherwise blend cap and wall normals. For a thin glass
    cylinder this turns its flat faces into a curved lens during refraction.
    Cylinder primitives are authored along local Z before their object pose.
    """
    import trimesh

    triangles = np.asarray(vertices)[faces]
    face_normals, valid = trimesh.triangles.normals(triangles)
    if not np.all(valid):
        raise ValueError("Cylinder contains degenerate triangles")
    corners = triangles.reshape(-1, 3)
    normals = np.zeros_like(corners, dtype=float)
    normals[:, :2] = corners[:, :2]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-15)
    cap_faces = np.abs(face_normals[:, 2]) > 1.0 - 1e-6
    cap_corners = np.repeat(cap_faces, 3)
    cap_normals = np.repeat(face_normals, 3, axis=0)
    normals[cap_corners] = cap_normals[cap_corners]
    # Retain the original triangle order, winding, vertex positions and UVs.
    new_uvs = np.asarray(uvs)[faces].reshape(-1, 2) if len(uvs) else uvs
    return corners, np.arange(len(corners)).reshape(-1, 3), normals, new_uvs


def _sharp_rigid_mesh_attributes(vertices, faces, uvs):
    """Smooth rigid mesh normals within 30-degree patches, preserving corners.

    This handles compound primitive meshes as well as authored rigid meshes.
    Triangle positions/order/winding remain unchanged; only rendering vertices
    are duplicated to represent discontinuous normals at a sharp edge.
    """
    import trimesh

    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    shaded = trimesh.graph.smooth_shade(source, angle=np.deg2rad(30.), facet_minarea=None)
    components = shaded.metadata.get("original_components")
    original_indices = np.hstack(components) if components is not None else np.arange(len(faces))
    triangles = np.asarray(vertices)[faces]
    if not np.array_equal(shaded.vertices[shaded.faces], triangles[original_indices]):
        raise ValueError("Sharp-edge shading unexpectedly changed rigid mesh triangles")
    corner_normals = np.empty_like(triangles, dtype=float)
    corner_normals[original_indices] = shaded.vertex_normals[shaded.faces]
    corners = triangles.reshape(-1, 3)
    new_uvs = np.asarray(uvs)[faces].reshape(-1, 2) if len(uvs) else uvs
    return corners, np.arange(len(corners)).reshape(-1, 3), corner_normals.reshape(-1, 3), new_uvs


class LuisaTrajectoryRenderer:
    def __init__(self, config, spec, geometry, destination):
        from .renderer import _camera_pose

        self.config, self.spec, self.geometry = config, spec, geometry
        self.objects = {} if geometry is None else {item.id: item for item in geometry.objects}
        self.bindings = {} if spec is None else {binding.object_id: binding for binding in spec.bindings}
        camera = config.get("camera", {})
        projection = camera.get("projection", "perspective")
        if projection not in {"perspective", "orthographic"}:
            raise ValueError(f"Unsupported camera projection {projection!r}")
        options = config.get("renderer", {})
        if isinstance(options, str):
            options = {}
        self.width, self.height = int(camera.get("width", 640)), int(camera.get("height", 480))
        self.spp = int(os.environ.get("WORLDSOLVER_LUISA_SPP", options.get("spp", 32)))
        if self.spp < 1:
            raise ValueError("Luisa samples per pixel must be positive")
        self.denoise = bool(options.get("denoise", True))
        self.device_index = int(os.environ.get("LUISA_DEVICE_INDEX", "0"))
        self.lr, context = _load_binding(destination)
        lr = self.lr
        lr.init(
            context_path=str(context),
            context_id="worldsolver",
            backend="cuda",
            device_index=self.device_index,
            log_level=lr.LogLevel.WARNING,
        )
        self.scene = lr.create_scene()
        self.scene.init(
            lr.Render(
                name="worldsolver",
                spectrum=lr.SRGBSpectrum(),
                integrator=lr.WavePathIntegrator(
                    log_level=lr.LogLevel.WARNING,
                    enable_cache=True,
                    max_depth=int(options.get("max_depth", 8)),
                    rr_depth=3,
                    rr_threshold=0.95,
                ),
                clamp_normal=180.0,
            )
        )
        camera_arguments = dict(
            name="benchmark_camera",
            pose=lr.MatrixTransform(np.ascontiguousarray(_camera_pose(camera))),
            film=lr.Film(resolution=(self.width, self.height)),
            filter=lr.Filter(),
            spp=self.spp,
        )
        if projection == "orthographic":
            extras = importlib.import_module("WorldSolverLuisaExtras")
            half_height = float(camera.get("orthographic_height", camera.get("scale", 2.0))) / 2.0
            if half_height <= 0:
                raise ValueError("Orthographic height must be positive")
            self.camera = extras.OrthographicCamera(**camera_arguments, zoom=float(np.log2(half_height)))
        else:
            self.camera = lr.PinholeCamera(**camera_arguments, fov=float(camera.get("yfov_degrees", 45.0)))
        self.scene.update_camera(self.camera, self.denoise)
        self.shapes = {}
        self.surfaces = {}
        self.rigid_surface_attributes = {}
        self.warnings = set()
        background = config.get("luisa_environment_color", camera.get("background_rgb", [0.035, 0.05, 0.075]))
        emission = np.asarray(background) if "luisa_environment_color" in config else _linear_rgb(background)
        self.scene.update_environment(
            lr.Environment(name="benchmark_background", emission=lr.ColorTexture(color=emission))
        )
        self._add_lights(camera, options)

    def _add_lights(self, camera, options):
        import trimesh
        from .renderer import _camera_pose

        eye = np.asarray(camera.get("position", [3, 3, 2]), dtype=float)
        target = np.asarray(camera.get("look_at", [0, 0, 0]), dtype=float)
        extent = max(float(np.linalg.norm(eye - target)), 0.2)
        pose = _camera_pose(camera)
        right, up, backward = (pose[:3, i] for i in range(3))
        # Keep studio light geometry behind the camera even in orthographic
        # views. Explicit source-scene lights remain unchanged.
        lights = self.config.get(
            "luisa_lights",
            [
                {
                    "pos": eye + (backward * 0.3 + right * 0.5 + up * 0.45) * extent,
                    "color": (1.0, 0.96, 0.90),
                    "intensity": 120.0,
                    "radius": extent * 0.12,
                },
                {
                    "pos": eye + (backward * 0.3 - right * 0.5 + up * 0.2) * extent,
                    "color": (0.75, 0.85, 1.0),
                    "intensity": 45.0,
                    "radius": extent * 0.12,
                },
            ],
        )
        for index, light in enumerate(lights):
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(light["radius"]))
            mesh.apply_translation(light["pos"])
            name = f"lighting_{index}"
            emission = self.lr.Light(
                name=name + "_emission",
                emission=self.lr.ColorTexture(color=np.asarray(light["color"]) * float(light["intensity"])),
                two_sided=True,
            )
            self.scene.update_emission(emission)
            shape = self.lr.DeformableShape(name=name, emission=emission)
            shape.update(
                vertices=np.ascontiguousarray(mesh.vertices),
                triangles=np.ascontiguousarray(mesh.faces),
                normals=np.ascontiguousarray(mesh.vertex_normals),
                uvs=np.empty((0, 2)),
            )
            self.scene.update_shape(shape)
            self.shapes[name] = shape

    def _surface(self, primitive, rgba=None, object_id=None, scalar_texture=None):
        material = primitive.material
        color = np.asarray(material.baseColorFactor if rgba is None else rgba, dtype=float)
        roughness, metallic = float(material.roughnessFactor), float(material.metallicFactor)
        overrides = self.config.get("render_materials", {}).get(object_id, {})
        binding = self.bindings.get(object_id)
        named_colors = {
            "green_felt": [0.06, 0.32, 0.14],
            "ivory": [0.94, 0.91, 0.81],
            "red": [0.78, 0.035, 0.025],
            "blue": [0.04, 0.22, 0.8],
            "white": [0.9, 0.9, 0.9],
            "black": [0.03, 0.03, 0.03],
        }
        if binding is not None and binding.color_rgb is None and binding.material in named_colors:
            color = np.asarray([*named_colors[binding.material], color[3]])
        if "color_rgb" in overrides:
            color = np.asarray([*overrides["color_rgb"], color[3]])
        roughness = float(overrides.get("roughness", roughness))
        texture = getattr(material, "baseColorTexture", None)
        source = None if texture is None else np.asarray(texture.source)
        if "texture_asset" in overrides:
            source = imageio.imread(overrides["texture_asset"])
        if scalar_texture is not None:
            # Diagnostic RGB values are linear vertex-color multipliers in
            # pyrender's mesh shader, not sRGB image pixels. Retain that
            # convention and keep the ramp fixed while only UVs change.
            source = scalar_texture
            color = np.asarray([1.0, 1.0, 1.0, binding.opacity if binding.opacity is not None else 1.0])
        texture_encoding = "linear" if scalar_texture is not None else "srgb"
        key_data = (
            color.tobytes()
            + np.asarray([roughness, metallic]).tobytes()
            + json.dumps(overrides, sort_keys=True).encode()
            + texture_encoding.encode()
        )
        if source is not None:
            key_data += source.tobytes()
        key = hashlib.sha256(key_data).hexdigest()[:20]
        if key in self.surfaces:
            return self.surfaces[key]
        lr = self.lr
        source_linear_colors = self.config.get("extension", {}).get("origin") == "Text2Sim"
        diffuse = lr.ColorTexture(color=color[:3] if overrides or source_linear_colors else _linear_rgb(color[:3]))
        opacity = lr.ColorTexture(color=(float(color[3]),))
        if source is not None:
            source = np.asarray(source, dtype=np.uint8)
            diffuse = lr.ImageTexture(
                image_data=np.ascontiguousarray(source[..., :3]).tobytes(),
                width=source.shape[1],
                height=source.shape[0],
                channel=3,
                scale=(1.0, 1.0, 1.0) if "texture_asset" in overrides else color[:3],
                encoding=texture_encoding,
            )
            if source.shape[-1] == 4:
                alpha = np.ascontiguousarray(source[..., 3:4])
                opacity = lr.ImageTexture(
                    image_data=alpha.tobytes(),
                    width=alpha.shape[1],
                    height=alpha.shape[0],
                    channel=1,
                    scale=(float(color[3]),) * 3,
                    encoding="linear",
                )
        # Official glass preset is transparent pale blue, low roughness, no
        # metallic component. Use a refractive BSDF rather than opaque plastic.
        is_glass = overrides.get("type") == "glass" or (binding is not None and binding.material == "glass")
        if is_glass:
            surface = lr.GlassSurface(
                name="surface_" + key,
                roughness=lr.ColorTexture(color=(max(roughness, 0.015),)),
                ks=lr.ColorTexture(color=(1.0, 1.0, 1.0)),
                kt=lr.ColorTexture(color=color[:3]),
                eta=lr.ColorTexture(color=(float(overrides.get("ior", 1.5)),)),
            )
        else:
            surface = lr.DisneySurface(
                name="surface_" + key,
                kd=diffuse,
                opacity=opacity,
                roughness=lr.ColorTexture(color=(roughness,)),
                metallic=lr.ColorTexture(color=(metallic,)),
                eta=lr.ColorTexture(color=(1.5,)),
                specular_trans=lr.ColorTexture(color=(0.0,)),
            )
        self.scene.update_surface(surface)
        self.surfaces[key] = surface
        return surface

    def _mesh(self, name, vertices, faces, normals, uvs, surface):
        shape = self.shapes.get(name)
        if shape is None:
            shape = self.lr.DeformableShape(name=name, surface=surface, clamp_normal=180.0)
        shape.update(
            vertices=np.ascontiguousarray(vertices, dtype=np.float64),
            triangles=np.ascontiguousarray(faces, dtype=np.uint32),
            normals=np.ascontiguousarray(normals, dtype=np.float64),
            uvs=np.ascontiguousarray(uvs, dtype=np.float64),
        )
        self.scene.update_shape(shape)
        self.shapes[name] = shape

    def render(self, frame):
        import trimesh
        from .renderer import _configure_opengl_platform, _add_bound_scene, _add_reconstructed_bindings, _texture_colors

        _configure_opengl_platform()
        import pyrender

        collector = _MeshCollector()
        if self.spec is not None and self.geometry is not None:
            for binding in self.spec.bindings:
                collector.object_id = binding.object_id
                one_spec = self.spec.model_copy(update={"bindings": [binding]})
                _add_bound_scene(collector, frame, self.geometry, one_spec, pyrender, trimesh)
                _add_reconstructed_bindings(collector, frame, self.geometry, one_spec, pyrender, trimesh)
        else:
            colors = _texture_colors(frame)
            collector.add(pyrender.Mesh.from_points(frame.position, colors=colors))
        used = set()
        object_mesh_counts = {}
        for mesh_index, (mesh, pose, object_id) in enumerate(collector.meshes):
            slot = object_mesh_counts.get(object_id, 0)
            object_mesh_counts[object_id] = slot + 1
            for primitive_index, primitive in enumerate(mesh.primitives):
                base_name = f"mesh_{object_id}_{slot}_{primitive_index}"
                if primitive.mode == 0:  # Scientific point displays: sphere instances, same centers.
                    colors = primitive.color_0
                    if colors is None:
                        colors = np.tile([0.15, 0.55, 0.95, 1.0], (len(primitive.positions), 1))
                    colors = np.rint(np.asarray(colors) * 255).astype(np.uint8)
                    unique, inverse = np.unique(colors, axis=0, return_inverse=True)
                    radius = float(self.config.get("camera", {}).get("point_world_radius", 0.005))
                    for color_index, color in enumerate(unique):
                        name = f"{base_name}_points_{color.tobytes().hex()}"
                        points = np.asarray(primitive.positions)[inverse == color_index]
                        points = trimesh.transform_points(points, pose)
                        surface = self._surface(primitive, color / 255.0, object_id)
                        shape = self.shapes.get(name)
                        if shape is None:
                            shape = self.lr.ParticlesShape(name=name, subdivision=1, surface=surface)
                        shape.update(centers=np.ascontiguousarray(points), radii=np.asarray([radius]))
                        self.scene.update_shape(shape)
                        self.shapes[name] = shape
                        used.add(name)
                    continue
                if primitive.mode != 4:
                    raise ValueError(f"Unsupported primitive mode {primitive.mode}; select pyrender")
                vertices = np.asarray(primitive.positions)
                faces = (
                    np.arange(len(vertices)).reshape(-1, 3)
                    if primitive.indices is None
                    else np.asarray(primitive.indices, dtype=np.int64).reshape(-1, 3)
                )
                normals = primitive.normals
                if normals is None:
                    normals = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).vertex_normals
                uvs = np.empty((0, 2)) if primitive.texcoord_0 is None else primitive.texcoord_0
                item = self.objects.get(object_id)
                if item is not None and self.config.get("render_materials", {}).get(object_id, {}).get("texture_asset"):
                    topology_path = item.geometry.topology_asset or item.geometry.asset
                    if topology_path and Path(topology_path).suffix == ".npz":
                        with np.load(topology_path) as topology:
                            if "uv" not in topology:
                                raise ValueError(f"Textured object {object_id} has no topology uv array")
                            uvs = topology["uv"]
                        if len(uvs) != len(vertices):
                            raise ValueError(f"UV/vertex count mismatch for {object_id}")
                scalar_texture = None
                binding = self.bindings.get(object_id)
                if (
                    item is not None
                    and binding is not None
                    and binding.method == "deformed_mesh"
                    and binding.scalar_attribute is not None
                    and binding.scalar_color_low is not None
                ):
                    # Apply even to a currently uniform field: later frames
                    # can then change colors through the same shape/material.
                    uvs, scalar_texture = _scalar_attribute_texture(frame, item, binding, len(vertices))
                if item is not None and item.geometry.type == "box":
                    # Shared corner normals make a planar tabletop look bent
                    # under path tracing. Keep its triangles and use the exact
                    # planar face normals, without moving any physical point.
                    triangles = vertices[faces]
                    normals = np.repeat(trimesh.triangles.normals(triangles)[0], 3, axis=0)
                    if len(uvs):
                        uvs = uvs[faces].reshape(-1, 2)
                    vertices = triangles.reshape(-1, 3)
                    faces = np.arange(len(vertices)).reshape(-1, 3)
                elif item is not None and item.geometry.type == "cylinder":
                    vertices, faces, normals, uvs = _cylinder_surface_attributes(vertices, faces, uvs)
                elif (
                    item is not None
                    and item.geometry.type == "triangle_mesh"
                    and binding is not None
                    and binding.method == "rigid_mesh"
                ):
                    # Rigid authoritative asset geometry never changes between
                    # frames; only its pose does. Split normals once per slot.
                    if base_name not in self.rigid_surface_attributes:
                        self.rigid_surface_attributes[base_name] = _sharp_rigid_mesh_attributes(vertices, faces, uvs)
                    vertices, faces, normals, uvs = self.rigid_surface_attributes[base_name]
                transforms = [pose] if primitive.poses is None else [pose @ p for p in primitive.poses]
                color = None
                if primitive.color_0 is not None and scalar_texture is None:
                    # A uniform vertex color can be represented exactly by a material.
                    colors = np.asarray(primitive.color_0)
                    if not np.allclose(colors, colors[0]):
                        raise ValueError(
                            "Luisa supports deformed-mesh scalar_attribute colors with explicit "
                            "linear low/high endpoints; other varying vertex colors require pyrender"
                        )
                    color = colors[0]
                surface = self._surface(primitive, color, object_id, scalar_texture=scalar_texture)
                for instance_index, transform in enumerate(transforms):
                    name = f"{base_name}_{instance_index}"
                    world_vertices = trimesh.transform_points(vertices, transform)
                    normal_matrix = np.linalg.inv(transform[:3, :3]).T
                    world_normals = normals @ normal_matrix.T
                    world_normals /= np.maximum(np.linalg.norm(world_normals, axis=1, keepdims=True), 1e-12)
                    self._mesh(name, world_vertices, faces, world_normals, uvs, surface)
                    used.add(name)
        # Luisa retains named shapes. A topology that temporarily vanishes must
        # not leave its previous frame visible. Collapse unused mesh slots far
        # outside the scene. This changes display only, never physical data.
        for name in set(self.shapes) - used:
            if name.startswith("lighting_"):
                continue
            shape = self.shapes[name]
            if "_points_" in name:
                shape.update(centers=np.asarray([[1e10, 1e10, 1e10]]), radii=np.asarray([1e-8]))
            else:
                shape.update(
                    vertices=np.full((3, 3), 1e10),
                    triangles=np.asarray([[0, 1, 2]]),
                    normals=np.tile([0.0, 0.0, 1.0], (3, 1)),
                    uvs=np.empty((0, 2)),
                )
            self.scene.update_shape(shape)
        self.scene.update_scene(time=float(frame.time))
        pixels = self.scene.render_frame(camera=self.camera)
        return np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)[..., :3]

    def close(self):
        # Camera owns OptiX denoiser and device buffers; release these before
        # destroying the CUDA context, otherwise its finalizer aborts Python.
        self.camera = None
        self.shapes.clear()
        self.surfaces.clear()
        self.rigid_surface_attributes.clear()
        self.lr.destroy()


def render_trajectory_luisa(trajectory, config, destination, *, spec=None, geometry=None, asset_root=None):
    import copy
    from .time_window import selected_video_frames, video_window

    config = copy.deepcopy(config)
    for material in config.get("render_materials", {}).values():
        if "texture_asset" in material and asset_root is not None:
            material["texture_asset"] = str((Path(asset_root) / material["texture_asset"]).resolve())
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    renderer = LuisaTrajectoryRenderer(config, spec, geometry, destination)
    frame_count = 0
    first_time = last_time = None
    try:
        with imageio.get_writer(
            destination,
            fps=int(config.get("camera", {}).get("fps", 30)),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            for frame in selected_video_frames(TrajectoryReader(trajectory), config):
                image = renderer.render(frame)
                writer.append_data(image)
                if frame_count == 0:
                    imageio.imwrite(destination.with_suffix(".first_frame.png"), image)
                    first_time = float(frame.time)
                last_time = float(frame.time)
                frame_count += 1
                if frame_count == 1 or frame_count % 30 == 0:
                    print(f"Luisa rendered {frame_count} frames in {time.monotonic() - start:.1f}s", flush=True)
    finally:
        renderer.close()
    report = {
        "backend": "luisa",
        "integrator": "WavePathIntegrator",
        "frame_count": frame_count,
        "width": renderer.width,
        "height": renderer.height,
        "spp": renderer.spp,
        "denoise": renderer.denoise,
        "device_index": renderer.device_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fps": int(config.get("camera", {}).get("fps", 30)),
        "first_physical_time": first_time,
        "last_physical_time": last_time,
        "configured_video_window": video_window(config),
        "elapsed_seconds": time.monotonic() - start,
        "geometry_source": "official WorldSolver authoritative mesh builders",
        "timing": "one trajectory state per frame; original benchmark fps preserved",
        "lighting": (
            "source-configured spherical lights and environment"
            if "luisa_lights" in config
            else "two camera-relative soft area lights and configured background"
        ),
        "projection": config.get("camera", {}).get("projection", "perspective"),
        "genesis_simulator_imported": "genesis" in sys.modules,
    }
    destination.with_suffix(".render.json").write_text(json.dumps(report, indent=2) + "\n")
