"""Check preservation of broad face structure while suppressing scan noise."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_refine import regional_target


class RegionalBlendChecks(unittest.TestCase):
    def test_broad_shape_is_retained_without_full_scan_noise(self):
        x, y = np.meshgrid(np.arange(-80., 82., 2.), np.arange(-65., 67., 2.))
        xy = np.c_[x.ravel(), y.ravel()]
        anchors = {'left_eye': [34., -30., 0.], 'right_eye': [34., 30., 0.],
                   'nose': [0., 0., 0.], 'mouth': [-40., 0., 0.]}
        broad = 4*np.exp(-.5*((xy[:, 0]+35)/22)**2-.5*(xy[:, 1]/25)**2)
        noise = np.random.default_rng(17).normal(0., 1., len(xy))
        source = broad+noise
        original = source.copy()
        target, detail = regional_target(xy, source, np.zeros(len(xy)), anchors)
        clean, _ = regional_target(xy, broad, np.zeros(len(xy)), anchors)
        baseline = .1*source
        self.assertLess(np.mean((target-broad)**2), np.mean((baseline-broad)**2))
        self.assertLess(np.std(target-clean), .30*np.std(noise))
        self.assertTrue(np.isfinite(target).all())
        self.assertTrue(np.array_equal(source, original))
        self.assertAlmostEqual(detail['retention_high_frequency'], .1)


if __name__ == '__main__':
    unittest.main()
