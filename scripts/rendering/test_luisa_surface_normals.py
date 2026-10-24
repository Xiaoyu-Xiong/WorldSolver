"""CPU-only geometry checks for native path-tracer shading attributes."""
import unittest

import numpy as np
import trimesh

from scripts.rendering.luisa_renderer import _cylinder_surface_attributes, _sharp_rigid_mesh_attributes


class CylinderNormalsTest(unittest.TestCase):
    def test_thin_glass_keeps_triangles_uvs_and_planar_caps(self):
        mesh = trimesh.creation.cylinder(radius=.566, height=.008, sections=32)
        uv = np.column_stack((np.arange(len(mesh.vertices)), -np.arange(len(mesh.vertices)))).astype(float)
        v, f, n, new_uv = _cylinder_surface_attributes(mesh.vertices, mesh.faces, uv)
        np.testing.assert_array_equal(v[f], mesh.vertices[mesh.faces])
        np.testing.assert_array_equal(new_uv[f], uv[mesh.faces])
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1., atol=1e-14)
        cap = np.abs(mesh.face_normals[:, 2]) > .999
        np.testing.assert_allclose(n[f][cap], np.repeat(mesh.face_normals[cap, None], 3, axis=1), atol=1e-14)
        self.assertTrue(np.all(np.abs(n[f][~cap, :, 2]) < 1e-14))
        # A rim location belongs to both a flat face and the round wall, with
        # different normals even though the geometric position is unchanged.
        rim = np.array([.566, 0., .004])
        matching = np.linalg.norm(v - rim, axis=1) < 1e-12
        directions = n[matching]
        self.assertTrue(np.any(np.abs(directions[:, 2] - 1.) < 1e-14))
        self.assertTrue(np.any(np.abs(directions[:, 0] - 1.) < 1e-14))

    def test_regular_cylinder_without_uvs_has_radial_side_normals(self):
        mesh = trimesh.creation.cylinder(radius=.2, height=1., sections=48)
        v, f, n, uv = _cylinder_surface_attributes(mesh.vertices, mesh.faces, np.empty((0, 2)))
        np.testing.assert_array_equal(v[f], mesh.vertices[mesh.faces])
        self.assertEqual(len(uv), 0)
        side = np.repeat(np.abs(mesh.face_normals[:, 2]) < .1, 3)
        self.assertTrue(np.all(np.einsum('ij,ij->i', n[side, :2], v[side, :2]) > 0))
        np.testing.assert_allclose(np.linalg.norm(n[side], axis=1), 1., atol=1e-14)


class RigidMeshNormalsTest(unittest.TestCase):
    def test_box_mesh_keeps_geometry_and_has_flat_faces(self):
        mesh = trimesh.creation.box(extents=[.712, .016, .118])
        uv = np.column_stack((mesh.vertices[:, 0], mesh.vertices[:, 2]))
        v, f, n, new_uv = _sharp_rigid_mesh_attributes(mesh.vertices, mesh.faces, uv)
        np.testing.assert_array_equal(v[f], mesh.vertices[mesh.faces])
        np.testing.assert_array_equal(new_uv[f], uv[mesh.faces])
        np.testing.assert_allclose(n[f], np.repeat(mesh.face_normals[:, None], 3, axis=1), atol=1e-14)

    def test_compound_rotor_faces_unchanged_and_box_walls_flat(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        mesh = trimesh.load_mesh(root / "scaffolds/172_text2sim_cloth_tumble_dryer/assets/rotor_visual.obj", process=False)
        v, f, n, uv = _sharp_rigid_mesh_attributes(mesh.vertices, mesh.faces, np.empty((0, 2)))
        np.testing.assert_array_equal(v[f], mesh.vertices[mesh.faces])
        self.assertEqual(len(uv), 0)
        # Exported rotor's first 30 parts are twelve-triangle planar wall boxes.
        np.testing.assert_allclose(n[f][:360], np.repeat(mesh.face_normals[:360, None], 3, axis=1), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1., atol=1e-12)

    def test_smooth_sphere_is_not_flat_shaded(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=.1)
        v, f, n, _ = _sharp_rigid_mesh_attributes(mesh.vertices, mesh.faces, np.empty((0, 2)))
        np.testing.assert_array_equal(v[f], mesh.vertices[mesh.faces])
        radial = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.assertGreater(float(np.min(np.einsum('ij,ij->i', n, radial))), .999)
        self.assertGreater(float(np.max(np.linalg.norm(n[f][:, 0] - n[f][:, 1], axis=1))), .05)


if __name__ == "__main__":
    unittest.main()
