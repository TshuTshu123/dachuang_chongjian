import unittest

import numpy as np
import open3d as o3d

from mesh_refine import bilateral_mesh_filter, observation_support


def plane_mesh():
    n = 20
    x, y = np.meshgrid(np.linspace(-.02, .02, n), np.linspace(-.02, .02, n))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            triangles.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(triangles))
    return mesh


class BilateralTests(unittest.TestCase):
    def test_exact_plane_and_topology_unchanged(self):
        mesh = plane_mesh()
        output, _ = bilateral_mesh_filter(mesh)
        np.testing.assert_allclose(np.asarray(mesh.vertices), np.asarray(output.vertices), atol=1e-12)
        np.testing.assert_array_equal(np.asarray(mesh.triangles), np.asarray(output.triangles))

    def test_noise_decreases_cap_and_boundaries_hold(self):
        mesh = plane_mesh()
        xyz = np.asarray(mesh.vertices)
        xyz[:, 2] = np.random.default_rng(6).normal(0, .00025, len(xyz))
        before = xyz.copy()
        output, report = bilateral_mesh_filter(mesh, max_move=.0002)
        after = np.asarray(output.vertices)
        self.assertLess(np.mean(after[:, 2] ** 2), np.mean(before[:, 2] ** 2))
        self.assertLessEqual(np.max(np.linalg.norm(after - before, axis=1)), .0002 + 1e-12)
        boundary = (np.abs(before[:, 0]) >= .01999) | (np.abs(before[:, 1]) >= .01999)
        np.testing.assert_array_equal(after[boundary], before[boundary])
        np.testing.assert_array_equal(np.asarray(mesh.vertices), before)
        self.assertGreater(report['frozen_boundary_or_nonmanifold_vertices'], 0)
        triangles = np.asarray(mesh.triangles)
        a, b = before[triangles], after[triangles]
        na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
        nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
        self.assertTrue((np.einsum('ij,ij->i', na, nb) > 0).all())

    def test_sharp_cube_is_preserved(self):
        mesh = o3d.geometry.TriangleMesh.create_box(.01, .01, .01)
        output, _ = bilateral_mesh_filter(mesh)
        np.testing.assert_allclose(np.asarray(output.vertices), np.asarray(mesh.vertices), atol=1e-12)

    def test_zero_weight_region_is_unchanged(self):
        mesh = plane_mesh()
        xyz = np.asarray(mesh.vertices)
        xyz[:, 2] = np.random.default_rng(8).normal(0, .00025, len(xyz))
        before = xyz.copy()
        weight = (xyz[:, 0] > 0).astype(float)
        output, _ = bilateral_mesh_filter(mesh, active_weights=weight)
        after = np.asarray(output.vertices)
        np.testing.assert_array_equal(after[weight == 0], before[weight == 0])
        self.assertGreater(np.linalg.norm(after[weight == 1] - before[weight == 1]), 0)

    def test_support_rejects_unobserved_surface(self):
        mesh = plane_mesh()
        mesh.compute_vertex_normals()
        _, labels, _, _, _ = observation_support(mesh, np.asarray(mesh.vertices))
        self.assertTrue((labels == 0).any())
        far = np.asarray(mesh.vertices).copy() + [0, 0, .03]
        _, labels, _, _, _ = observation_support(mesh, far)
        self.assertTrue((labels == 2).all())


if __name__ == '__main__':
    unittest.main()
