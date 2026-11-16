"""Controlled comparison: original scan / uniform FLAME / regional residual blend."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_refine import load_model, fit_template, make_mesh, patch
from face_pipeline import remap_parts
from body_parts import file_hash
from ply_to_obj import convert_ply_to_obj
from tools.flame_face_trial import comparison


def trial(scan):
    start = time.perf_counter()
    previous = json.loads((ROOT/'outputs/face_v26_validation/previous_models.json').read_text('utf-8'))
    source = ROOT/previous[str(scan)]['source']
    digest = file_hash(source)
    data = json.loads((source.parent/'body_parts.json').read_text('utf-8'))
    if digest != previous[str(scan)]['sha256'] or digest != data['mesh']['sha256']:
        raise ValueError('Original scan labels/hash mismatch')
    mesh = o3d.io.read_triangle_mesh(str(source))
    mesh.compute_vertex_normals()
    template, basis, faces, model_hash = load_model()
    fitted, nose, fit = fit_template(mesh, data['parts'], template, basis, faces)
    reference = make_mesh(fitted, faces)
    attempts, selected = [], None
    for radius in [1., .9, .8]:
        try:
            uniform, uniform_ids, uniform_check = patch(mesh, reference, data['parts'], nose, radius_scale=radius)
            regional, ids, check = patch(mesh, reference, data['parts'], nose, radius_scale=radius,
                                         style='regional', anchors=fit['region_anchors'])
            record = {'radius_scale': radius, 'uniform': uniform_check, 'regional': check}
            attempts.append(record)
            if uniform_check['geometry_checks_passed'] and check['geometry_checks_passed']:
                labels, _ = remap_parts(data, mesh, regional, ids)
                assert np.array_equal(np.asarray(uniform.triangles), np.asarray(regional.triangles))
                assert np.array_equal(uniform_ids, ids)
                selected = (uniform, regional, ids, labels, record)
                break
        except (ValueError, RuntimeError) as error:
            attempts.append({'radius_scale': radius, 'reason': str(error)})
    out = ROOT/'outputs/experiments/optimization_13'/f'export_{scan:05d}'
    out.mkdir(parents=True, exist_ok=True)
    report = {'scan': scan, 'source': str(source), 'source_sha256': digest, 'model_sha256': model_hash,
              'fit': fit, 'attempts': attempts, 'applied': selected is not None,
              'same_source_fit_and_face_connectivity': selected is not None}
    if selected:
        uniform, regional, ids, labels, record = selected
        comparison([mesh, uniform, regional], ['原扫描', '统一融合', '分区融合试验'], nose, out/'face_comparison.jpg')
        for name, candidate in [('uniform', uniform), ('regional', regional)]:
            target = out/(name+'_mesh.ply')
            if not o3d.io.write_triangle_mesh(str(target), candidate):
                raise IOError('Export failed')
            readback = o3d.io.read_triangle_mesh(str(target))
            assert np.array_equal(np.asarray(readback.vertices), np.asarray(candidate.vertices))
            assert np.array_equal(np.asarray(readback.triangles), np.asarray(candidate.triangles))
        convert_ply_to_obj(out/'regional_mesh.ply', out/'regional_mesh.obj', vertex_colors=True)
        labels['mesh'].update(file='regional_mesh.ply', sha256=file_hash(out/'regional_mesh.ply'),
                              vertex_count=len(regional.vertices), triangle_count=len(regional.triangles))
        labels['fingerprint']['mesh_sha256'] = labels['mesh']['sha256']
        labels.setdefault('notes', []).append('Regional FLAME trial: original labels mapped by explicit indices; new face labels assigned by model.')
        (out/'body_parts.json').write_text(json.dumps(labels, ensure_ascii=False, separators=(',', ':')), 'utf-8')
        np.savez_compressed(out/'vertex_mapping.npz', source_vertex_index=ids)
        report['selected'] = record
        report['output_sha256'] = labels['mesh']['sha256']
    report.update(source_unchanged=file_hash(source) == digest, seconds=time.perf_counter()-start)
    (out/'face_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({'scan': scan, 'applied': report['applied'], 'output': str(out),
                      'seconds': report['seconds'], 'selected': report.get('selected', {}).get('radius_scale')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    for value in parser.parse_args().scan:
        trial(value)
