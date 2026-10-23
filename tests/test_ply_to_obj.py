import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from ply_to_obj import convert_ply_to_obj


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.ply'
        self.mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector([[0, 0, 0], [2, 0, 0], [0, 3, 0]]),
            o3d.utility.Vector3iVector([[0, 1, 2]]))
        self.mesh.compute_vertex_normals()
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(np.eye(3))

    def test_binary_and_ascii_mesh_roundtrip(self):
        for ascii_mode in (False, True):
            with self.subTest(ascii=ascii_mode):
                o3d.io.write_triangle_mesh(str(self.source), self.mesh, write_ascii=ascii_mode)
                result = convert_ply_to_obj(self.source, overwrite=True)
                loaded = o3d.io.read_triangle_mesh(str(result['output']))
                np.testing.assert_allclose(loaded.vertices, self.mesh.vertices)
                np.testing.assert_array_equal(loaded.triangles, self.mesh.triangles)
                np.testing.assert_allclose(loaded.vertex_normals, self.mesh.vertex_normals)
                self.assertFalse(result['vertex_colors'])

    def test_color_extension_and_parent_directory(self):
        o3d.io.write_triangle_mesh(str(self.source), self.mesh)
        result = convert_ply_to_obj(self.source, self.root / 'obj' / 'color.obj', vertex_colors=True)
        lines = result['output'].read_text().splitlines()
        vertices = [line.split() for line in lines if line.startswith('v ')]
        self.assertTrue(result['vertex_colors'])
        self.assertEqual(vertices[0], ['v', '0', '0', '0', '1', '0', '0'])
        self.assertIn('f 1//1 2//2 3//3', lines)

    def test_point_cloud_requires_explicit_mode(self):
        cloud = o3d.geometry.PointCloud(self.mesh.vertices)
        o3d.io.write_point_cloud(str(self.source), cloud)
        with self.assertRaisesRegex(ValueError, 'no faces'):
            convert_ply_to_obj(self.source)
        self.assertFalse(self.source.with_suffix('.obj').exists())
        result = convert_ply_to_obj(self.source, points_only=True)
        self.assertEqual(result['faces'], 0)
        self.assertIn('p 3', result['output'].read_text())

    def test_overwrite_is_explicit(self):
        o3d.io.write_triangle_mesh(str(self.source), self.mesh)
        result = convert_ply_to_obj(self.source)
        before = result['output'].read_bytes()
        with self.assertRaises(FileExistsError):
            convert_ply_to_obj(self.source)
        self.assertEqual(before, result['output'].read_bytes())

    def test_invalid_paths(self):
        with self.assertRaises(FileNotFoundError):
            convert_ply_to_obj(self.source)
        with self.assertRaises(ValueError):
            convert_ply_to_obj(self.root / 'input.obj')
        with self.assertRaises(ValueError):
            convert_ply_to_obj(self.source, self.source)
