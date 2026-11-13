"""检查正式整理步骤的禁用、缓存复用、受保护部位和发布失败回退。

python tools/check_scan_polish.py；真实几何收益另由完整 run.py 回归检查。
"""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import open3d as o3d
import run
from body_parts import file_hash
from scan_polish_pipeline import apply_scan_polish, publish_step


class PublishChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='polish-check-')
        self.folder = Path(self.temp.name)
        self.mesh = o3d.geometry.TriangleMesh.create_sphere(radius=50, resolution=8)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([.5, .5, .5])
        self.ply = self.folder/'sample_mesh.ply'
        o3d.io.write_triangle_mesh(str(self.ply), self.mesh)
        self.mesh = o3d.io.read_triangle_mesh(str(self.ply))
        self.obj = self.ply.with_suffix('.obj')
        run.convert_ply_to_obj(self.ply, self.obj, vertex_colors=True)
        self.data = {'mesh': {'file': self.ply.name, 'sha256': file_hash(self.ply)},
                     'fingerprint': {'mesh_sha256': file_hash(self.ply)}, 'parts': {
                         'upper_clothing': {'vertex_indices': list(range(len(self.mesh.vertices)))},
                         'left_hand': {'vertex_indices': []}}, 'notes': []}
        self.parts = self.folder/'body_parts.json'
        self.parts.write_text(json.dumps(self.data), encoding='utf-8')
        raw_source = self.folder/'sample.ply'
        raw_source.write_bytes(self.ply.read_bytes())
        self.report = {'source': str(raw_source), 'subject': {'kind': 'human'},
                       'segmentation': {'status': 'complete', 'sha256': file_hash(self.parts)},
                       'artifacts': {key: {'name': path.name, 'sha256': file_hash(path)}
                                     for key, path in (('ply', self.ply), ('obj', self.obj), ('base', self.ply))}}
        self.path = self.folder/'report.json'
        self.path.write_text(json.dumps(self.report), encoding='utf-8')
        self.result = {'status': 'complete', 'applied': True, 'method': 'test checked movement', 'topology_changed': False}
        self.candidate = copy.deepcopy(self.mesh)
        xyz = np.asarray(self.candidate.vertices).copy()
        xyz[0] *= 1.001
        self.candidate.vertices = o3d.utility.Vector3dVector(xyz)

    def tearDown(self):
        self.temp.cleanup()

    def test_completed_stage_is_not_smoothed_twice(self):
        with patch('scan_polish_pipeline.refine_torso', return_value=(self.candidate, self.result, self.data['parts'])) as refine:
            first = apply_scan_polish(self.path, False, True)
            digest = file_hash(self.ply)
            second = apply_scan_polish(self.path, False, True)
        self.assertEqual(refine.call_count, 1)
        self.assertTrue(first['torso_surface_refinement']['applied'])
        self.assertEqual(file_hash(self.ply), digest)
        self.assertEqual(second['artifacts']['ply']['sha256'], digest)
        data = json.loads(self.parts.read_text(encoding='utf-8'))
        self.assertEqual(data['mesh']['sha256'], digest)
        self.assertEqual(data['fingerprint']['mesh_sha256'], digest)

    def test_failed_transaction_restores_all_files(self):
        initial = {p: p.read_bytes() for p in (self.ply, self.obj, self.parts, self.path)}
        real_publish = run.publish_with_inherited_permissions
        count = 0
        def fail_once(source, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('simulated second-file publish failure')
            return real_publish(source, destination)
        with patch('run.publish_with_inherited_permissions', side_effect=fail_once):
            with self.assertRaises(OSError):
                publish_step(self.path, self.report, self.data, self.mesh, self.candidate,
                             self.data['parts'], self.result, 'torso_surface_refinement')
        for p, content in initial.items():
            self.assertEqual(p.read_bytes(), content, str(p))

    def test_protected_hand_movement_is_rejected(self):
        self.data['parts']['left_hand']['vertex_indices'] = [0]
        self.parts.write_text(json.dumps(self.data), encoding='utf-8')
        digest = file_hash(self.ply)
        with patch('scan_polish_pipeline.refine_torso', return_value=(self.candidate, self.result, self.data['parts'])):
            report = apply_scan_polish(self.path, False, True)
        self.assertEqual(report['torso_surface_refinement']['status'], 'failed')
        self.assertIn('Protected region moved', report['torso_surface_refinement']['reason'])
        self.assertEqual(file_hash(self.ply), digest)

    def test_object_and_disabled_stages_do_not_call_geometry(self):
        self.report['subject']['kind'] = 'object'
        self.path.write_text(json.dumps(self.report), encoding='utf-8')
        with patch('scan_polish_pipeline.refine_torso') as refine:
            report = apply_scan_polish(self.path, False, True)
        refine.assert_not_called()
        self.assertEqual(report['mano_surface_refinement']['status'], 'disabled')
        self.assertEqual(report['torso_surface_refinement']['status'], 'skipped')
        self.assertEqual(file_hash(self.ply), self.report['artifacts']['ply']['sha256'])


if __name__ == '__main__':
    unittest.main()
