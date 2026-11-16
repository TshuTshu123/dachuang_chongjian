"""Verify model switches, CLI priority, cache keys and pipeline routing."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from settings import resolve_model_settings


class ModelSettingsChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='model-settings-')
        self.folder = Path(self.temp.name)
        self.settings = self.folder/'settings.json'
        self.source = self.folder/'sample.ply'
        self.source.write_text('routing fixture', encoding='utf-8')
        self.report_path = self.folder/'report.json'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, hand, face):
        self.settings.write_text(json.dumps({'use_mano_hand_model': hand, 'use_flame_face_model': face}), 'utf-8')

    def route(self, **kwargs):
        report = {'source': str(self.source), 'subject': {'kind': 'human'}, 'artifacts': {'ply': {'name': 'mesh.ply'}}}
        with ExitStack() as stack:
            stack.enter_context(patch('run.fingerprint', return_value={'version': 26}))
            cache = stack.enter_context(patch('run.cached_result', return_value=(self.report_path, report)))
            for name in ['add_body_part_positions', 'add_hand_refinement', 'add_surface_quality',
                         'add_skeleton_hand_refinement', 'add_unified_hand_refinement']:
                stack.enter_context(patch('run.'+name, return_value=report))
            mano = stack.enter_context(patch('run.apply_mano', return_value=report))
            polish = stack.enter_context(patch('run.apply_scan_polish', return_value=report))
            face = stack.enter_context(patch('run.apply_face_refinement', return_value=report))
            stack.enter_context(patch('run.finalize_model_files', return_value=report))
            stack.enter_context(patch('run.export_face_uv', return_value={'status': 'disabled'}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run.reconstruct(self.source, raw_color=False, settings_path=self.settings, **kwargs)
        return mano.call_args.args[1], face.call_args.args[1], polish.call_args.args[1], cache.call_args.args[1]

    def test_independent_switches_reach_pipeline_and_cache(self):
        keys = set()
        for hand, face in [(False, False), (False, True), (True, False), (True, True)]:
            with self.subTest(hand=hand, face=face):
                self.write(hand, face)
                actual_hand, actual_face, ordinary_hand, signature = self.route()
                self.assertEqual((actual_hand, actual_face), (hand, face))
                self.assertTrue(ordinary_hand)
                self.assertEqual(signature['model_application'], {'mano': hand, 'flame': face})
                saved = json.loads(self.report_path.read_text('utf-8'))
                self.assertEqual(saved['model_settings']['effective'], {'mano': hand, 'flame': face})
                keys.add(json.dumps(signature, sort_keys=True))
        self.assertEqual(len(keys), 4)

    def test_explicit_overrides_and_processing_disable(self):
        self.write(False, False)
        self.assertEqual(self.route(hand_model=True, face_model=True)[:2], (True, True))
        self.assertEqual(self.route(hand_model=True, face_model=True, hand_auto=False, face_refine=False)[:2], (False, False))
        self.assertEqual(self.route(hand_model=True, face_model=True, segmentation=False)[:2], (False, False))

    def test_invalid_settings_fail_before_processing(self):
        for value in [{'use_mano_hand_model': 'false'}, {'use_flame_face_model': 0}, {'use_mano': True}, []]:
            self.settings.write_text(json.dumps(value), 'utf-8')
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_model_settings(self.settings)
        self.settings.write_text('{}', 'utf-8-sig')
        self.assertTrue(resolve_model_settings(self.settings)['requested']['use_mano_hand_model'])

    def test_cli_default_defers_to_settings_and_flags_override(self):
        with patch('run.select_inputs', return_value=[self.source]), patch('run.reconstruct') as reconstruct:
            run.main(['0'])
            self.assertIsNone(reconstruct.call_args.kwargs['hand_model'])
            self.assertIsNone(reconstruct.call_args.kwargs['face_model'])
            run.main(['0', '--no-hand-model', '--face-model', '--settings', str(self.settings)])
            self.assertFalse(reconstruct.call_args.kwargs['hand_model'])
            self.assertTrue(reconstruct.call_args.kwargs['face_model'])
            self.assertEqual(reconstruct.call_args.kwargs['settings_path'], str(self.settings))


if __name__ == '__main__':
    unittest.main()
