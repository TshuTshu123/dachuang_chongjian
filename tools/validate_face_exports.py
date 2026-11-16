"""Independently check v26 PLY/OBJ, semantic labels, mappings and old files."""
from pathlib import Path
import json
import sys

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from body_parts import file_hash


def check(scan, folder, expected_version=26):
    entry = json.loads((folder/f'export_{scan:05d}.json').read_text('utf-8'))
    path = Path(entry['report'])
    report = json.loads(path.read_text('utf-8'))
    assert report['fingerprint']['version'] == expected_version
    assert file_hash(report['source']) == report['fingerprint']['source_sha256']
    for name, digest in report['fingerprint']['code_sha256'].items():
        assert file_hash(ROOT/name) == digest, name
    for key, artifact in report['artifacts'].items():
        assert file_hash(path.parent/artifact['name']) == artifact['sha256'], key
    source = path.parent/report['artifacts']['ply']['name']
    mesh = o3d.io.read_triangle_mesh(str(source))
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    assert np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v)
    obj_v, obj_f = [], []
    with (path.parent/report['artifacts']['obj']['name']).open('r', encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('v '):
                obj_v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                obj_f.append([int(x.split('/')[0])-1 for x in line.split()[1:]])
    assert np.array_equal(np.asarray(obj_v), v), 'OBJ positions mismatch'
    assert np.array_equal(np.asarray(obj_f), f), 'OBJ connectivity mismatch'
    labels_ok = None
    if report.get('segmentation', {}).get('status') == 'complete':
        parts_path = path.parent/'body_parts.json'
        assert file_hash(parts_path) == report['segmentation']['sha256']
        data = json.loads(parts_path.read_text('utf-8'))
        assert data['mesh']['sha256'] == file_hash(source) == data['fingerprint']['mesh_sha256']
        assert data['mesh']['vertex_count'] == len(v) and data['mesh']['triangle_count'] == len(f)
        assigned = np.zeros(len(v), bool)
        for name, part in data['parts'].items():
            ids = np.asarray(part['vertex_indices'], int)
            assert len(ids) == part['vertex_count'] == len(np.unique(ids)), name
            if len(ids):
                assert ids.min() >= 0 and ids.max() < len(v) and not assigned[ids].any(), name
                assigned[ids] = True
                assert np.allclose(v[ids].mean(0), part['center_mm'], atol=1e-10, rtol=0), name
                assert np.array_equal(v[ids].min(0), part['bbox_mm']['min']), name
                assert np.array_equal(v[ids].max(0), part['bbox_mm']['max']), name
        assert int((~assigned).sum()) == data['unlabeled_vertex_count']
        labels_ok = True
    face = report['flame_face_refinement']
    mapping_ok = None
    if face['applied']:
        assert face['status'] == 'complete' and face['geometry_checks_passed']
        assert face['output_mesh_sha256'] == file_hash(source)
        artifact = face['vertex_mapping']
        p = path.parent/artifact['name']
        assert file_hash(p) == artifact['sha256']
        d = np.load(p)
        assert str(d['output_mesh_sha256']) == file_hash(source)
        assert str(d['source_mesh_sha256']) == face['source_mesh_sha256']
        to_old, to_new = d['source_vertex_index'], d['old_to_new']
        retained = to_old >= 0
        assert len(to_old) == len(v) and np.array_equal(to_new[to_old[retained]], np.flatnonzero(retained))
        assert len(np.unique(to_old[retained])) == int(retained.sum())
        assert face['selected']['intersections_touching_new_patch'] == 0
        assert face['selected']['patch_winding_conflicts'] == 0
        mapping_ok = True
    if (path.parent/'hand_centerlines.json').exists():
        centers = path.parent/'hand_centerlines.json'
        d = json.loads(centers.read_text('utf-8'))
        assert d['output_mesh_sha256'] == file_hash(source)
        if report.get('hand_processing', {}).get('centerlines'):
            assert file_hash(centers) == report['hand_processing']['centerlines']['sha256']
    uv = report.get('face_uv', {})
    if uv.get('status') == 'complete':
        assert uv['input']['mesh_sha256'] == file_hash(source)
        for name, digest in uv['sha256'].items():
            assert file_hash(path.parent/'textured'/name) == digest
    topology = report['final_topology']
    assert topology.get('vertices') == len(v) and topology.get('triangles') == len(f)
    result = {'scan': scan, 'report': str(path), 'ply': str(source), 'sha256': file_hash(source),
              'vertices': len(v), 'triangles': len(f), 'ply_obj_identical': True,
              'labels_valid': labels_ok, 'vertex_mapping_valid': mapping_ok,
              'face_applied': face['applied'], 'face_status': face['status'],
              'selected_radius': face.get('selected', {}).get('radius_scale'),
              'uv_status': uv.get('status'), 'source_and_code_hashes_match': True}
    (path.parent/'face_export_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    return result


def main():
    folder = ROOT/'outputs/face_v26_validation'
    results = []
    for scan in range(6):
        result = check(scan, folder)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    previous = json.loads((folder/'previous_models.json').read_text('utf-8'))
    for data in previous.values():
        assert file_hash(ROOT/data['source']) == data['sha256']
    output = {'exports': results, 'previous_models_unchanged': True}
    (folder/'final_validation.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), 'utf-8')


if __name__ == '__main__':
    main()
