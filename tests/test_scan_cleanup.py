"""Scan preprocessing regression tests using synthetic disconnected surfaces."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

from surface_detail import select_main_point_component
from reconstruct_human import parse_args


class MainComponentTests(unittest.TestCase):
    def test_background_removed_colors_and_input_preserved(self):
        x, y = np.meshgrid(np.arange(20) * .004, np.arange(20) * .004)
        body = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        background = body[:60] + [1., 0., 0.]
        xyz = np.vstack([body, background])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        rgb = np.random.default_rng(0).random(xyz.shape)
        cloud.colors = o3d.utility.Vector3dVector(rgb)
        kept, removed, report = select_main_point_component(cloud, .012)
        self.assertEqual(len(kept.points), len(body))
        self.assertEqual(len(removed.points), len(background))
        self.assertEqual(report['removed_points'], len(background))
        np.testing.assert_allclose(np.asarray(kept.points), body)
        np.testing.assert_array_equal(np.asarray(kept.colors), rgb[:len(body)])
        np.testing.assert_array_equal(np.asarray(cloud.points), xyz)

    def test_no_supported_cluster_fails_with_actionable_message(self):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            np.column_stack([np.arange(10), np.zeros((10, 2))])))
        with self.assertRaisesRegex(RuntimeError, 'point-components all'):
            select_main_point_component(cloud, .012)


class PresetTests(unittest.TestCase):
    def arguments(self, *flags):
        # This file is only checked for existence; no geometry is read.
        with patch('sys.argv', ['reconstruct_human.py', str(Path(__file__)), *flags]):
            return parse_args()

    def test_existing_presets_keep_their_behavior(self):
        args = self.arguments('--preset', 'detail')
        self.assertEqual(args.depth, 10)
        self.assertEqual(args.voxel, .001)
        self.assertEqual(args.point_denoise_iterations, 0)
        self.assertEqual(args.smooth_iterations, 0)
        self.assertEqual(args.point_components, 'all')

    def test_scan_and_explicit_overrides(self):
        args = self.arguments('--preset', 'scan')
        self.assertEqual(args.depth, 9)
        self.assertEqual(args.point_denoise_iterations, 2)
        self.assertEqual(args.smooth_iterations, 2)
        self.assertEqual(args.point_components, 'largest')
        self.assertEqual(args.max_extrapolation_mm, 0)
        args = self.arguments('--preset', 'scan', '--depth', '10',
                              '--point-denoise-iterations', '0', '--smooth-iterations', '0',
                              '--point-components', 'all')
        self.assertEqual(args.depth, 10)
        self.assertEqual(args.point_denoise_iterations, 0)
        self.assertEqual(args.smooth_iterations, 0)
        self.assertEqual(args.point_components, 'all')


if __name__ == '__main__':
    unittest.main()
