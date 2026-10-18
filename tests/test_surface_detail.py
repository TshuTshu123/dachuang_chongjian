"""Synthetic geometry checks; no Open3D, scan input, or mesh reconstruction.

Run from the project root: python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from surface_detail import denoise_local_planes, fit_local_planes


class LocalPlaneTests(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.linspace(-0.012, 0.012, 25),
                           np.linspace(-0.012, 0.012, 25))
        self.plane = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])

    def test_plane_normals_and_zero_bias(self):
        normal, offset, valid, flat, _ = fit_local_planes(self.plane, 24, 0.006, 1)
        self.assertTrue(valid.all())
        self.assertTrue(flat.all())
        np.testing.assert_allclose(np.abs(normal[:, 2]), 1, atol=1e-10)
        np.testing.assert_allclose(offset, 0, atol=1e-12)
        filtered, _ = denoise_local_planes(self.plane, 24, 0.006, 1, 2)
        np.testing.assert_allclose(filtered, self.plane, atol=1e-12)

    def test_noise_reduction_and_total_cap(self):
        noisy = self.plane.copy()
        noisy[:, 2] = np.random.default_rng(4).normal(0, 0.00025, len(noisy))
        original = noisy.copy()
        filtered, report = denoise_local_planes(noisy, 24, 0.006, 1, 3, 0.0002)
        self.assertLess(np.sqrt(np.mean(filtered[:, 2] ** 2)),
                        np.sqrt(np.mean(noisy[:, 2] ** 2)))
        self.assertLessEqual(np.linalg.norm(filtered - noisy, axis=1).max(), 0.0002 + 1e-12)
        self.assertLessEqual(report['movement_max_mm'], 0.2 + 1e-9)
        np.testing.assert_array_equal(noisy, original)

    def test_sparse_and_collinear_points_are_not_projected(self):
        sparse = self.plane[::50]
        filtered, report = denoise_local_planes(sparse, 12, 0.0001, 1, 1)
        np.testing.assert_array_equal(filtered, sparse)
        self.assertEqual(report['moved_points'], 0)
        line = np.column_stack([np.linspace(0, 0.01, 20), np.zeros((20, 2))])
        _, _, valid, flat, _ = fit_local_planes(line, 12, 0.006, 1)
        self.assertFalse(valid.any())
        self.assertFalse(flat.any())

    def test_batch_independence(self):
        a = fit_local_planes(self.plane, 24, 0.006, 1, batch_size=17)
        b = fit_local_planes(self.plane, 24, 0.006, 1, batch_size=300)
        for left, right in zip(a, b):
            np.testing.assert_allclose(left, right, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
