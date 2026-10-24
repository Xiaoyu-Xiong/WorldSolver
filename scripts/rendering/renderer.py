from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from worldsolver.config import load_yaml
from worldsolver.contracts import StateFrame
from worldsolver.contracts.scene_geometry import SceneGeometry, SceneObject, load_scene_geometry
from worldsolver.trajectory import TrajectoryReader

from .render_spec import RenderSpec, load_render_spec
from .video_writer import write_video
from .time_window import selected_video_frames


def _configure_opengl_platform() -> None:
    """Select EGL automatically for headless Unix before importing PyOpenGL."""
    # Compatibility for the last PyPI pyrender release when used with NumPy 2.
    if not hasattr(np, "infty"):
        np.infty = np.inf  # type: ignore[attr-defined]
    if "PYOPENGL_PLATFORM" not in os.environ and os.name != "nt" and not os.environ.get("DISPLAY"):
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def _camera_pose(camera: dict) -> np.ndarray:
    eye = np.asarray(camera.get("position", [3, 3, 2]), dtype=float)
    target = np.asarray(camera.get("look_at", [0, 0, 0]), dtype=float)
    world_up = np.asarray(camera.get("up", [0, 0, 1]), dtype=float)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:
        fallback = (
            np.asarray([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.asarray([1.0, 0.0, 0.0])
        )
        right = np.cross(forward, fallback)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, up, -forward))
    pose[:3, 3] = eye
    return pose


def _body_transform(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = _quaternion_matrix(quaternion)
    transform[:3, 3] = position
    return transform


def _material(pyrender, name: str, color=None, alpha: float = 1.0):
    presets = {
        "glass": ([0.78, 0.90, 0.98], 0.18, 0.0, 0.08),
        "red_wine": ([0.34, 0.004, 0.018], 0.96, 0.0, 0.18),
        "metal": ([0.55, 0.58, 0.62], 1.0, 0.86, 0.2),
        "ceramic": ([0.82, 0.80, 0.74], 1.0, 0.0, 0.22),
        "rubber": ([0.12, 0.13, 0.15], 1.0, 0.0, 0.82),
        "wood": ([0.38, 0.18, 0.07], 1.0, 0.0, 0.58),
        "matte": ([0.42, 0.48, 0.56], 1.0, 0.0, 0.7),
        "default": ([0.15, 0.55, 0.95], 1.0, 0.0, 0.5),
    }
    preset_base, preset_alpha, metallic, roughness = presets.get(name, presets["default"])
    base = np.asarray(color if color is not None else preset_base, dtype=float)
    final_alpha = (
        max(float(alpha), float(preset_alpha))
        if name == "glass"
        else float(alpha) * float(preset_alpha)
    )
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.append(np.clip(base, 0.0, 1.0), final_alpha),
        metallicFactor=metallic,
        roughnessFactor=roughness,
        alphaMode="BLEND" if final_alpha < 0.999 else "OPAQUE",
        doubleSided=True,
    )


def _object_pose(item: SceneObject) -> np.ndarray:
    pose = _body_transform(
        np.asarray(item.transform.translation, dtype=float),
        np.asarray(item.transform.rotation_xyzw, dtype=float),
    )
    pose[:3, :3] *= np.asarray(item.transform.scale, dtype=float)
    return pose


def _binding_material(pyrender, binding, color=None):
    """Apply explicit appearance without changing historical material presets."""
    if binding.color_rgb is not None:
        color = binding.color_rgb
    material = _material(pyrender, binding.material, color)
    if binding.opacity is not None:
        rgba = np.array(material.baseColorFactor, copy=True)
        rgba[3] = binding.opacity
        material.baseColorFactor = rgba
        material.alphaMode = "BLEND" if binding.opacity < 0.999 else "OPAQUE"
    return material


def _diagnostic_colors(frame, subset, item, binding):
    """Sample an explicitly declared scalar, never infer physics from particle density."""
    values = None
    if binding.scalar_attribute:
        if binding.scalar_attribute not in subset.attributes:
            raise ValueError(f"Missing color attribute: {binding.scalar_attribute}")
        values = np.asarray(subset.attributes[binding.scalar_attribute], dtype=float)
        if values.shape != (len(subset.position),):
            raise ValueError("Color attribute must have one scalar per sample")
    elif binding.scalar_field:
        from scipy.ndimage import map_coordinates

        if binding.scalar_field not in frame.fields:
            raise ValueError(f"Missing color field: {binding.scalar_field}")
        field = np.asarray(frame.fields[binding.scalar_field], dtype=float)
        if field.ndim != len(binding.field_axes):
            raise ValueError("Color field dimensions must match field_axes")
        low = np.asarray(item.geometry.bounds_min, dtype=float)
        high = np.asarray(item.geometry.bounds_max, dtype=float)
        axes = binding.field_axes
        if np.any(high[axes] <= low[axes]):
            raise ValueError("Color field bounds must have positive extent")
        coords = [
            (subset.position[:, axis] - low[axis]) / (high[axis] - low[axis]) * n - 0.5
            for axis, n in zip(axes, field.shape, strict=True)
        ]
        values = map_coordinates(field, coords, order=1, mode="nearest")
    if values is not None:
        if not np.isfinite(values).all():
            raise ValueError("Color scalar contains non-finite values")
        lo, hi = binding.scalar_range
        t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        if binding.scalar_color_low is not None:
            low = np.asarray(binding.scalar_color_low)
            high = np.asarray(binding.scalar_color_high)
            return low[None, :] + t[:, None] * (high - low)[None, :]
        # Fixed blue-white-red ramp: equal colors always mean equal scalar values.
        anchors = np.array([[0.12, 0.30, 0.80], [0.94, 0.94, 0.90], [0.82, 0.12, 0.08]])
        return np.column_stack([np.interp(t, [0.0, 0.5, 1.0], anchors[:, c]) for c in range(3)])
    if binding.color_rgb is not None:
        return np.tile(binding.color_rgb, (len(subset.position), 1))
    return _texture_colors(subset)


def _axisymmetric_mesh(trimesh, item: SceneObject):
    profile = np.asarray(item.geometry.inner_profile, dtype=float)
    wall = float(item.geometry.wall_thickness)
    outer = profile.copy()
    outer[:, 0] += wall
    cross_section = np.vstack((outer, profile[::-1]))
    sections = 96
    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    vertices = np.asarray(
        [
            [radius * np.cos(angle), radius * np.sin(angle), z]
            for radius, z in cross_section
            for angle in angles
        ]
    )
    faces: list[tuple[int, int, int]] = []
    for row in range(len(cross_section) - 1):
        for column in range(sections):
            nxt = (column + 1) % sections
            a, b = row * sections + column, row * sections + nxt
            c, d = (row + 1) * sections + nxt, (row + 1) * sections + column
            faces.extend(((a, b, c), (a, c, d)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def _authoritative_mesh(item: SceneObject, trimesh):
    geometry = item.geometry
    if geometry.type == "box":
        return trimesh.creation.box(extents=2.0 * np.asarray(geometry.half_extents))
    if geometry.type == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=float(geometry.radius))
    if geometry.type == "cylinder":
        return trimesh.creation.cylinder(
            radius=float(geometry.radius), height=float(geometry.height)
        )
    if geometry.type == "capsule":
        return trimesh.creation.capsule(
            radius=float(geometry.radius), height=float(geometry.height)
        )
    if geometry.type == "axisymmetric_container":
        return _axisymmetric_mesh(trimesh, item)
    if geometry.type in {"triangle_mesh", "heightfield"} and geometry.asset:
        return trimesh.load_mesh(geometry.asset)
    return None


def _binding_indices(frame: StateFrame, item: SceneObject) -> np.ndarray:
    binding = item.state_binding
    if binding is None:
        return np.empty(0, dtype=int)
    if binding.selector == "all":
        return np.arange(len(frame.position), dtype=int)
    if binding.selector == "body_id":
        if frame.body_id is None:
            return np.empty(0, dtype=int)
        return np.flatnonzero(np.asarray(frame.body_id) == binding.value)
    if binding.selector == "attribute":
        values = frame.attributes.get(binding.attribute_name)
        if values is None:
            return np.empty(0, dtype=int)
        return np.flatnonzero(np.asarray(values).reshape(-1) == binding.value)
    return np.empty(0, dtype=int)


def _frame_subset(frame: StateFrame, indices: np.ndarray) -> StateFrame:
    return StateFrame(
        frame.time,
        np.asarray(frame.position)[indices],
        np.asarray(frame.velocity)[indices],
        orientation=None if frame.orientation is None else np.asarray(frame.orientation)[indices],
        angular_velocity=(
            None if frame.angular_velocity is None else np.asarray(frame.angular_velocity)[indices]
        ),
        mass=None if frame.mass is None else np.asarray(frame.mass)[indices],
        body_id=None if frame.body_id is None else np.asarray(frame.body_id)[indices],
        attributes={name: np.asarray(value)[indices] for name, value in frame.attributes.items()},
        fields=frame.fields,
        scalars=frame.scalars,
    )


def _load_topology(item: SceneObject, trimesh):
    asset = item.geometry.topology_asset or item.geometry.asset
    if not asset:
        return None
    path = Path(asset)
    if path.suffix.casefold() == ".npz":
        data = np.load(path)
        if "faces" in data:
            return trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)
        return {"vertices": data["vertices"], "edges": data["edges"]}
    if path.suffix.casefold() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "faces" in data:
            return trimesh.Trimesh(
                vertices=np.asarray(data["vertices"]),
                faces=np.asarray(data["faces"]),
                process=False,
            )
        return {
            "vertices": np.asarray(data["vertices"]),
            "edges": np.asarray(data["edges"]),
        }
    return trimesh.load_mesh(asset, process=False)


def _add_rigid_binding(scene, frame, item, binding, pyrender, trimesh) -> None:
    mesh = _authoritative_mesh(item, trimesh)
    if mesh is None:
        return
    if binding.wireframe:
        # Keep sharp physical edges, not triangulation diagonals across the smoke.
        edges = mesh.face_adjacency_edges[mesh.face_adjacency_angles > 1e-4]
        radius = max(float(np.min(np.maximum(mesh.extents, 1e-6))) * 0.0015, 0.0005)
        if len(edges):
            mesh = trimesh.util.concatenate(
                [
                    trimesh.creation.cylinder(
                        segment=mesh.vertices[edge], radius=radius, sections=6
                    )
                    for edge in edges
                ]
            )
    indices = _binding_indices(frame, item)
    material = _binding_material(pyrender, binding)
    if item.motion == "static" or len(indices) == 0:
        pose = _object_pose(item)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True), pose=pose)
        return
    # Render every bound body as an instanced mesh, not just the first one.
    orientation = (
        np.asarray(frame.orientation)
        if frame.orientation is not None
        else np.tile(np.asarray(item.transform.rotation_xyzw), (len(frame.position), 1))
    )
    positions = np.asarray(frame.position)[indices]
    rotations = np.asarray([_quaternion_matrix(q) for q in orientation[indices]])
    scale = np.asarray(item.transform.scale, dtype=float)
    poses = np.zeros((len(indices), 4, 4), dtype=float)
    poses[:, 3, 3] = 1.0
    poses[:, :3, :3] = rotations * scale
    poses[:, :3, 3] = positions
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, poses=poses, smooth=True))


def _add_deformed_binding(scene, frame, item, binding, pyrender, trimesh) -> None:
    topology = _load_topology(item, trimesh)
    indices = _binding_indices(frame, item)
    if topology is None or len(indices) != len(topology.vertices):
        return
    subset = _frame_subset(frame, indices)
    topology.vertices = np.asarray(subset.position)
    vertex_colors = _diagnostic_colors(frame, subset, item, binding)
    if binding.scalar_attribute is not None:
        alpha = binding.opacity if binding.opacity is not None else 1.0
        rgba = np.column_stack((vertex_colors, np.full(len(vertex_colors), alpha)))
        topology.visual.vertex_colors = np.rint(rgba * 255).astype(np.uint8)
        render_mesh = pyrender.Mesh.from_trimesh(topology, smooth=True)
    else:
        material = _binding_material(pyrender, binding)
        render_mesh = pyrender.Mesh.from_trimesh(topology, material=material, smooth=True)
    scene.add(render_mesh)


def _curve_segments(item: SceneObject, trimesh):
    topology = _load_topology(item, trimesh)
    if topology is None:
        return None
    if isinstance(topology, dict):
        return np.asarray(topology["edges"], dtype=int)
    if hasattr(topology, "edges_unique"):
        return np.asarray(topology.edges_unique, dtype=int)
    return None


def _add_curve_binding(scene, frame, item, binding, pyrender, trimesh) -> None:
    segments = _curve_segments(item, trimesh)
    indices = _binding_indices(frame, item)
    if segments is None or len(indices) == 0:
        return
    points = np.asarray(frame.position)[indices]
    # A curve-network topology must describe the sampled state points.  Older
    # 080 rod scaffolds stored only the two geometric endpoints, while the
    # trajectory contains the full 65-point centerline.  Recover the intended
    # ordered chain for that legacy endpoint-only asset so intermediate
    # connections are not silently omitted from the render.
    if len(points) > 2 and len(segments) == 1 and np.array_equal(segments[0], [0, 1]):
        segments = np.column_stack(
            (np.arange(len(points) - 1), np.arange(1, len(points)))
        )
    material = _binding_material(pyrender, binding)
    radius = float(item.geometry.radius or 0.005)
    for left, right in segments:
        if left >= len(points) or right >= len(points):
            continue
        delta = points[right] - points[left]
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            continue
        cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=10)
        direction = delta / length
        rotation = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
        pose = rotation if rotation is not None else np.eye(4)
        pose[:3, 3] = 0.5 * (points[left] + points[right])
        scene.add(pyrender.Mesh.from_trimesh(cylinder, material=material), pose=pose)


def _add_particle_instances(scene, frame, item, binding, pyrender, trimesh) -> None:
    indices = _binding_indices(frame, item)
    if len(indices) == 0:
        return
    points = np.asarray(frame.position)[indices]
    radii = np.asarray(frame.attributes.get("particle_radius", np.full(len(frame.position), 0.02)))[
        indices
    ]
    colors = _texture_colors(_frame_subset(frame, indices))
    if colors is None:
        colors = np.tile([[0.5, 0.5, 0.5]], (len(points), 1))
    # Quantized material batches avoid one draw call per grain while preserving scientific scale.
    radius = float(np.median(radii)) * binding.particle_radius_scale
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    poses = np.tile(np.eye(4), (len(points), 1, 1))
    poses[:, :3, 3] = points
    material = _binding_material(pyrender, binding, np.median(colors, axis=0))
    scene.add(pyrender.Mesh.from_trimesh(sphere, material=material, poses=poses, smooth=True))


def _volume_slice_rgba(density, binding, step):
    """Beer-Lambert alpha for one cell-thick slab; zero density stays transparent."""
    color = binding.color_rgb if binding.color_rgb is not None else [0.72, 0.76, 0.82]
    alpha = -np.expm1(-np.maximum(density, 0.0) * binding.density_scale * step)
    if binding.opacity is not None:
        alpha *= binding.opacity
    rgba = np.empty(density.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = np.rint(np.asarray(color) * 255).astype(np.uint8)
    rgba[..., 3] = np.rint(np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    return rgba


def _volume_point_rgba(samples, binding):
    """Apply each volume binding's scientific channel color and optical density."""
    samples = np.maximum(np.asarray(samples, dtype=float), 0.0)
    alpha = -np.expm1(-samples * binding.density_scale)
    if binding.opacity is not None:
        alpha *= binding.opacity
    color = binding.color_rgb if binding.color_rgb is not None else [0.72, 0.76, 0.82]
    return np.column_stack((np.tile(color, (len(samples), 1)), np.clip(alpha, 0.0, 0.85)))


def _add_volume_slices(scene, density, item, binding, pyrender):
    """Opt-in textured slabs for orthographic smoke inspection, not point thresholding."""
    from PIL import Image

    low = np.asarray(item.geometry.bounds_min, dtype=float)
    high = np.asarray(item.geometry.bounds_max, dtype=float)
    axis = binding.volume_axis
    other = [q for q in range(3) if q != axis]
    step = (high[axis] - low[axis]) / density.shape[axis]
    for k in range(density.shape[axis]):
        slab = np.take(density, k, axis=axis)
        # Texture upload flips image rows; PIL row zero is the top of the slab.
        rgba = _volume_slice_rgba(slab.T[::-1], binding, step)
        if not np.any(rgba[..., 3]):
            continue
        vertices = np.zeros((4, 3), dtype=np.float32)
        vertices[:, axis] = low[axis] + (k + 0.5) * step
        vertices[:, other[0]] = [low[other[0]], high[other[0]], high[other[0]], low[other[0]]]
        vertices[:, other[1]] = [low[other[1]], low[other[1]], high[other[1]], high[other[1]]]
        texture = pyrender.Texture(
            source=Image.fromarray(rgba),
            source_channels="RGBA",
            sampler=pyrender.Sampler(magFilter=9729, minFilter=9729, wrapS=33071, wrapT=33071),
        )
        material = pyrender.MetallicRoughnessMaterial(
            # Some pyrender versions do not detect texture-only transparency.
            baseColorFactor=[1.0, 1.0, 1.0, 0.999],
            baseColorTexture=texture,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="BLEND",
            doubleSided=True,
        )
        center = vertices.mean(axis=0)
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal /= np.linalg.norm(normal)
        primitive = pyrender.Primitive(
            positions=vertices - center,
            normals=np.tile(normal, (4, 1)),
            texcoord_0=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            indices=np.array([[0, 1, 2], [0, 2, 3]]),
            material=material,
        )
        # The renderer sorts transparent meshes by node translation, not mesh bounds.
        pose = np.eye(4)
        pose[:3, 3] = center
        scene.add(pyrender.Mesh(primitives=[primitive]), pose=pose)


def _scalar_grid_samples(frame, item, binding):
    """A fixed cell-center diagnostic grid avoids holes in advected marker clouds."""
    if binding.scalar_field not in frame.fields:
        raise ValueError(f"Missing color field: {binding.scalar_field}")
    field = np.asarray(frame.fields[binding.scalar_field])
    if field.ndim != 2:
        raise ValueError("scalar_grid requires a two-dimensional field")
    low = np.asarray(item.geometry.bounds_min, dtype=float)
    high = np.asarray(item.geometry.bounds_max, dtype=float)
    axes = binding.field_axes
    coords = [
        low[a] + (np.arange(n) + 0.5) * (high[a] - low[a]) / n
        for a, n in zip(axes, field.shape, strict=True)
    ]
    grid = np.meshgrid(*coords, indexing="ij")
    points = np.tile(0.5 * (low + high), (field.size, 1))
    for a, values in zip(axes, grid, strict=True):
        points[:, a] = values.ravel()
    mask = frame.fields.get("active_mask")
    if mask is not None:
        if np.asarray(mask).shape != field.shape:
            raise ValueError("scalar_grid active_mask shape differs from scalar field")
        points = points[np.asarray(mask).astype(bool).ravel()]
    return StateFrame(time=frame.time, position=points, velocity=np.zeros_like(points))


def _add_volume_binding(scene, frame, item, binding, pyrender, trimesh) -> None:
    if binding.opacity == 0.0:
        return
    field_name = item.state_binding.attribute_name if item.state_binding is not None else None
    if not field_name or field_name not in frame.fields:
        return
    density = np.asarray(frame.fields[field_name], dtype=float)
    if density.ndim != 3 or float(density.max()) <= 0.0:
        return
    if binding.volume_mode == "slices":
        _add_volume_slices(scene, density, item, binding, pyrender)
        return
    threshold = max(float(np.percentile(density, 75)), float(density.max()) * 0.08)
    indices = np.argwhere(density >= threshold)
    if len(indices) > 12000:
        stride = int(np.ceil(len(indices) / 12000))
        indices = indices[::stride]
    low = np.asarray(item.geometry.bounds_min, dtype=float)
    high = np.asarray(item.geometry.bounds_max, dtype=float)
    scale = (high - low) / np.maximum(np.asarray(density.shape) - 1, 1)
    points = low + indices * scale
    colors = _volume_point_rgba(density[tuple(indices.T)], binding)
    scene.add(pyrender.Mesh.from_points(points, colors=colors))


def _add_bound_scene(
    scene,
    frame: StateFrame,
    geometry: SceneGeometry | None,
    spec: RenderSpec | None,
    pyrender,
    trimesh,
) -> None:
    if spec is None or geometry is None:
        return
    objects = {item.id: item for item in geometry.objects}
    for binding in spec.bindings:
        item = objects[binding.object_id]
        if not binding.visible:
            continue
        if binding.method == "rigid_mesh":
            _add_rigid_binding(scene, frame, item, binding, pyrender, trimesh)
        elif binding.method in {"deformed_mesh", "shell_surface"}:
            _add_deformed_binding(scene, frame, item, binding, pyrender, trimesh)
        elif binding.method == "curve_tubes":
            _add_curve_binding(scene, frame, item, binding, pyrender, trimesh)
        elif binding.method == "particle_instances":
            _add_particle_instances(scene, frame, item, binding, pyrender, trimesh)
        elif binding.method == "volume":
            _add_volume_binding(scene, frame, item, binding, pyrender, trimesh)


def _add_reconstructed_bindings(
    scene, frame, geometry: SceneGeometry, spec: RenderSpec, pyrender, trimesh
) -> None:
    objects = {item.id: item for item in geometry.objects}
    for binding in spec.bindings:
        if not binding.visible or binding.method not in {
            "reconstructed_surface",
            "diagnostic_points",
        }:
            continue
        item = objects[binding.object_id]
        subset = _frame_subset(frame, _binding_indices(frame, item))
        if binding.method == "reconstructed_surface":
            surface = _particle_surface(subset, binding, trimesh)
            if surface is None:
                continue
            colors = _texture_colors(subset)
            color = np.median(colors, axis=0) if colors is not None else None
            scene.add(
                pyrender.Mesh.from_trimesh(
                    surface,
                    material=_binding_material(pyrender, binding, color),
                    smooth=True,
                )
            )
        else:
            if binding.scalar_grid:
                subset = _scalar_grid_samples(frame, item, binding)
            if binding.max_points is not None and len(subset.position) > binding.max_points:
                indices = np.linspace(0, len(subset.position) - 1, binding.max_points, dtype=int)
                subset = _frame_subset(subset, indices)
            point_colors = _diagnostic_colors(frame, subset, item, binding)
            if point_colors is None:
                point_colors = np.tile([[0.15, 0.55, 0.95]], (len(subset.position), 1))
            alpha = binding.opacity if binding.opacity is not None else 1.0
            rgba = np.column_stack((point_colors, np.full(len(point_colors), alpha)))
            scene.add(pyrender.Mesh.from_points(subset.position, colors=rgba))


def _particle_surface(frame: StateFrame, binding, trimesh):
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    points = np.asarray(frame.position, dtype=float)
    if len(points) < 4:
        return None
    radii = np.asarray(frame.attributes.get("particle_radius", np.full(len(points), 0.02))).reshape(
        -1
    )
    radius = float(np.median(radii)) * binding.particle_radius_scale
    pitch = float(binding.surface_pitch or max(radius * 1.25, 1e-3))
    low = points.min(axis=0) - 3.0 * radius
    high = points.max(axis=0) + 3.0 * radius
    shape = np.maximum(np.ceil((high - low) / pitch).astype(int) + 1, 4)
    if int(np.prod(shape)) > 1_500_000:
        pitch *= (float(np.prod(shape)) / 1_500_000.0) ** (1.0 / 3.0)
        shape = np.maximum(np.ceil((high - low) / pitch).astype(int) + 1, 4)
    indices = np.clip(np.rint((points - low) / pitch).astype(int), 0, shape - 1)
    density = np.zeros(tuple(shape), dtype=np.float32)
    np.add.at(density, tuple(indices.T), 1.0)
    sigma = max(radius / pitch * binding.surface_smoothing, 0.65)
    density = gaussian_filter(density, sigma=sigma)
    peak = float(density.max())
    if peak <= 0.0:
        return None
    level = min(peak * 0.20, 0.09)
    vertices, faces, _, _ = marching_cubes(density, level=level, spacing=(pitch, pitch, pitch))
    vertices += low
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _create_opengl_scene(
    frame: StateFrame,
    camera: dict,
    config: dict | None = None,
    spec: RenderSpec | None = None,
    geometry: SceneGeometry | None = None,
):
    _configure_opengl_platform()
    import pyrender  # type: ignore[import-untyped]
    import trimesh

    background = np.asarray(camera.get("background_rgb", [0.035, 0.05, 0.075]), dtype=float)
    scene = pyrender.Scene(
        bg_color=np.append(np.clip(background, 0.0, 1.0), 1.0),
        ambient_light=np.asarray(camera.get("ambient_light", [0.22, 0.24, 0.28]), dtype=float),
    )
    width, height = int(camera.get("width", 640)), int(camera.get("height", 480))
    projection = camera.get("projection", "perspective")
    if projection == "orthographic":
        ymag = float(camera.get("orthographic_height", camera.get("scale", 2.0))) / 2.0
        camera_node = pyrender.OrthographicCamera(
            xmag=ymag * width / height, ymag=ymag, znear=0.01, zfar=1000.0
        )
    else:
        camera_node = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(float(camera.get("yfov_degrees", 45.0))), znear=0.01, zfar=1000.0
        )
    camera_pose = _camera_pose(camera)
    scene.add(camera_node, pose=camera_pose)

    key = pyrender.DirectionalLight(color=np.ones(3), intensity=float(camera.get("key_light", 3.0)))
    scene.add(key, pose=camera_pose)
    fill_pose = camera_pose.copy()
    fill_pose[:3, 3] = np.asarray(camera.get("fill_light_position", [-4.0, -2.0, 7.0]))
    scene.add(
        pyrender.PointLight(color=np.asarray([0.72, 0.82, 1.0]), intensity=18.0),
        pose=fill_pose,
    )
    _add_bound_scene(scene, frame, geometry, spec, pyrender, trimesh)

    if spec is not None and geometry is not None:
        _add_reconstructed_bindings(scene, frame, geometry, spec, pyrender, trimesh)
    else:
        point_colors = _texture_colors(frame)
        if point_colors is None:
            point_colors = np.tile(np.asarray([[0.15, 0.55, 0.95]]), (len(frame.position), 1))
        rgba = np.column_stack((np.clip(point_colors, 0.0, 1.0), np.ones(len(point_colors))))
        scene.add(pyrender.Mesh.from_points(np.asarray(frame.position), colors=rgba))

    return scene


def _render_opengl(
    frame: StateFrame,
    camera: dict,
    renderer=None,
    config: dict | None = None,
    spec=None,
    geometry=None,
) -> np.ndarray:
    _configure_opengl_platform()
    import pyrender  # type: ignore[import-untyped]

    width, height = int(camera.get("width", 640)), int(camera.get("height", 480))
    owns_renderer = renderer is None
    renderer = renderer or pyrender.OffscreenRenderer(
        viewport_width=width,
        viewport_height=height,
        point_size=float(camera.get("point_radius", 4)) * 2.0,
    )
    flags = pyrender.RenderFlags.RGBA
    if camera.get("shadows", True):
        flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    try:
        color, _ = renderer.render(
            _create_opengl_scene(frame, camera, config=config, spec=spec, geometry=geometry),
            flags=flags,
        )
        return np.ascontiguousarray(color[:, :, :3], dtype=np.uint8)
    finally:
        if owns_renderer:
            renderer.delete()


def _texture_colors(frame: StateFrame) -> np.ndarray | None:
    attributes = frame.attributes
    if "color_rgb" in attributes:
        return np.clip(np.asarray(attributes["color_rgb"], dtype=float), 0.0, 1.0)
    if "texture_rgb" in attributes:
        return np.clip(np.asarray(attributes["texture_rgb"], dtype=float), 0.0, 1.0)
    if "texture_uv" not in attributes:
        return None
    uv = np.asarray(attributes["texture_uv"], dtype=float)
    phase = 2.0 * np.pi * (5.0 * uv[:, 0] + 1.5 * np.sin(2.0 * np.pi * uv[:, 1]))
    colors = np.column_stack(
        (
            0.5 + 0.5 * np.sin(phase),
            0.5 + 0.5 * np.sin(phase + 2.094),
            0.5 + 0.5 * np.sin(phase + 4.189),
        )
    )
    if "density" in attributes:
        density = np.asarray(attributes["density"], dtype=float).reshape(-1, 1)
        density /= max(float(np.percentile(density, 99)), 1e-12)
        colors *= np.clip(0.15 + density, 0.0, 1.0)
    return np.clip(colors, 0.0, 1.0)


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = max(float(np.linalg.norm((x, y, z, w))), 1e-12)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def render_trajectory(
    trajectory: Path,
    config_path: Path,
    destination: Path,
    *,
    render_spec_path: Path | None = None,
    geometry_path: Path | None = None,
    backend: str | None = None,
) -> None:
    config = load_yaml(config_path)
    camera = config.get("camera", {})
    if render_spec_path is None:
        candidate = config_path.with_name("render_spec.yaml")
        render_spec_path = candidate if candidate.is_file() else None
    spec = load_render_spec(render_spec_path)
    if geometry_path is None:
        candidate = config_path.with_name("scene_geometry.yaml")
        geometry_path = candidate if candidate.is_file() else None
    geometry = load_scene_geometry(geometry_path)
    renderer_options = config.get("renderer", {})
    configured_backend = renderer_options if isinstance(renderer_options, str) else renderer_options.get("backend")
    backend = backend or os.environ.get("WORLDSOLVER_RENDERER") or configured_backend or "luisa"
    if backend == "luisa":
        from .luisa_renderer import render_trajectory_luisa

        render_trajectory_luisa(
            trajectory, config, destination, spec=spec, geometry=geometry, asset_root=config_path.parent
        )
        return
    if backend != "pyrender":
        raise ValueError(f"Unknown rendering backend {backend!r}; choose luisa or pyrender")
    _configure_opengl_platform()
    import pyrender  # type: ignore[import-untyped]

    renderer = pyrender.OffscreenRenderer(
        viewport_width=int(camera.get("width", 640)),
        viewport_height=int(camera.get("height", 480)),
        point_size=float(camera.get("point_radius", 4)) * 2.0,
    )
    try:
        frames = (
            _render_opengl(
                frame,
                camera,
                renderer=renderer,
                config=config,
                spec=spec,
                geometry=geometry,
            )
            for frame in selected_video_frames(TrajectoryReader(trajectory), config)
        )
        write_video(destination, frames, int(camera.get("fps", 30)))
    finally:
        renderer.delete()


def scene_render_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-spec", type=Path, default=None)
    parser.add_argument("--geometry", type=Path, default=None)
    parser.add_argument("--renderer", choices=("luisa", "pyrender"), default=None)
    arguments = parser.parse_args()
    render_trajectory(
        arguments.trajectory,
        arguments.config,
        arguments.output,
        render_spec_path=arguments.render_spec,
        geometry_path=arguments.geometry,
        backend=arguments.renderer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(scene_render_main())
