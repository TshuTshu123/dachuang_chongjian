"""Check FLAME publishing rollback, explicit label remapping and cache reuse."""
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
from face_pipeline import apply_face_refinement, publish_face, remap_parts


class FacePipelineChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='flame-check-')
        self.folder = Path(self.temp.name)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=50, resolution=12)
        mesh.paint_uniform_color([.5, .6, .7])
        mesh.compute_vertex_normals()
        self.ply = self.folder/'sample_mesh.ply'
        o3d.io.write_triangle_mesh(str(self.ply), mesh)
        self.original = o3d.io.read_triangle_mesh(str(self.ply))
        self.original.compute_vertex_normals()
        self.obj = self.ply.with_suffix('.obj')
        run.convert_ply_to_obj(self.ply, self.obj, vertex_colors=True)
        self.candidate = self.original.subdivide_midpoint(1)
        self.mapping = np.r_[np.arange(len(self.original.vertices)),
                             np.full(len(self.candidate.vertices)-len(self.original.vertices), -1)]
        self.data = {'status': 'complete', 'mesh': {'file': self.ply.name, 'sha256': file_hash(self.ply)},
                     'fingerprint': {'mesh_sha256': file_hash(self.ply)}, 'parts': {
                         'face_neck': {'vertex_indices': list(range(51, len(self.original.vertices)))},
                         'hair': {'vertex_indices': list(range(1, 51))},
                         'left_hand': {'vertex_indices': [0]}}, 'notes': []}
        self.parts = self.folder/'body_parts.json'
        self.parts.write_text(json.dumps(self.data), 'utf-8')
        source = self.folder/'sample.ply'
        source.write_bytes(self.ply.read_bytes())
        self.report = {'source': str(source), 'subject': {'kind': 'human'},
                       'segmentation': {'status': 'complete', 'sha256': file_hash(self.parts)},
                       'artifacts': {key: {'name': p.name, 'sha256': file_hash(p)}
                                     for key, p in [('ply', self.ply), ('obj', self.obj), ('base', source)]}}
        self.path = self.folder/'report.json'
        self.path.write_text(json.dumps(self.report), 'utf-8')
        self.result = {'status': 'complete', 'applied': True, 'method': 'checked subdivision fixture',
                       'geometry_checks_passed': True}

    def tearDown(self):
        self.temp.cleanup()

    def mocks(self, passed=True):
        return (patch('face_pipeline.load_model', return_value=(np.asarray(self.original.vertices),
                 np.zeros((len(self.original.vertices), 3, 20)), np.asarray(self.original.triangles), 'fixture')),
                patch('face_pipeline.fit_template', return_value=(np.asarray(self.original.vertices), np.zeros(3),
                 {'history': [{'median_correspondence_mm': 1., 'p90_correspondence_mm': 2.}]})),
                patch('face_pipeline.patch', return_value=(self.candidate, self.mapping,
                 {'geometry_checks_passed': passed})))

    def test_mapping_and_completed_cache(self):
        a, b, c = self.mocks()
        with a, b, c as geometry:
            result = apply_face_refinement(self.path, axis='x')
            digest = file_hash(self.ply)
            again = apply_face_refinement(self.path, axis='x')
        self.assertEqual(geometry.call_count, 1)
        self.assertTrue(result['flame_face_refinement']['applied'])
        self.assertEqual(again['artifacts']['ply']['sha256'], digest)
        self.assertEqual(file_hash(self.ply), digest)
        data = json.loads(self.parts.read_text('utf-8'))
        self.assertEqual(data['mesh']['sha256'], digest)
        self.assertEqual(data['parts']['left_hand']['vertex_indices'], [0])
        self.assertEqual(data['parts']['hair']['vertex_indices'], list(range(1, 51)))
        self.assertEqual(data['parts']['face_neck']['model_assigned_vertex_count'], int((self.mapping < 0).sum()))
        saved = np.load(self.folder/'intermediate/flame/vertex_mapping.npz')
        self.assertTrue(np.array_equal(saved['source_vertex_index'], self.mapping))

    def test_publish_failure_restores_all_files(self):
        initial = {p: p.read_bytes() for p in [self.ply, self.obj, self.parts, self.path]}
        real = run.publish_with_inherited_permissions
        count = 0
        def fail_once(source, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('simulated face publication failure')
            return real(source, destination)
        with patch('run.publish_with_inherited_permissions', side_effect=fail_once):
            with self.assertRaises(OSError):
                publish_face(self.path, self.report, self.data, self.original,
                             self.candidate, self.mapping, self.result)
        for p, content in initial.items():
            self.assertEqual(p.read_bytes(), content)
        self.assertFalse((self.folder/'intermediate/flame/vertex_mapping.npz').exists())

    def test_removing_protected_hand_is_rejected(self):
        mapping = self.mapping.copy()
        mapping[0] = -1
        with self.assertRaisesRegex(ValueError, 'protected part: left_hand'):
            remap_parts(self.data, self.original, self.candidate, mapping)

    def test_rejected_candidates_leave_input_and_are_not_repeated(self):
        digest = file_hash(self.ply)
        a, b, c = self.mocks(False)
        with a, b, c as geometry:
            result = apply_face_refinement(self.path, axis='x')
            apply_face_refinement(self.path, axis='x')
        self.assertEqual(geometry.call_count, 6)
        self.assertFalse(result['flame_face_refinement']['applied'])
        self.assertEqual(file_hash(self.ply), digest)

    def test_disabled_and_nonhuman_do_not_load_weights(self):
        with patch('face_pipeline.load_model') as load:
            result = apply_face_refinement(self.path, False)
            self.assertEqual(result['flame_face_refinement']['status'], 'disabled')
            report = copy.deepcopy(self.report)
            report['subject']['kind'] = 'object'
            self.path.write_text(json.dumps(report), 'utf-8')
            result = apply_face_refinement(self.path)
            self.assertEqual(result['flame_face_refinement']['status'], 'skipped')
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
