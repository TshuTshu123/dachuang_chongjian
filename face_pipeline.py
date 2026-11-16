"""FLAME 正式步骤：拟合、检查、标签重映射及整组原子发布。

在手部/躯干步骤后、最终文件及 UV 纹理导出前调用。失败保留该步输入。
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import open3d as o3d

from body_parts import file_hash
from ply_to_obj import convert_ply_to_obj
from face_refine import load_model, fit_template, make_mesh, patch

FIELD = 'flame_face_refinement'
FACE_PARTS = {'face_neck', 'upper_lip', 'lower_lip', 'upper_teeth', 'lower_teeth', 'tongue'}


def remap_parts(data, original, candidate, source_ids):
    old, xyz = np.asarray(original.vertices), np.asarray(candidate.vertices)
    source_ids = np.asarray(source_ids, int)
    existing = source_ids >= 0
    if len(source_ids) != len(xyz) or (existing.any() and source_ids[existing].max() >= len(old)):
        raise ValueError('Invalid face vertex mapping')
    if len(np.unique(source_ids[existing])) != existing.sum():
        raise ValueError('Duplicate retained source vertices')
    if not np.array_equal(xyz[existing], old[source_ids[existing]]):
        raise ValueError('Face step moved retained source vertices')
    if not np.array_equal(np.asarray(candidate.vertex_colors)[existing],
                          np.asarray(original.vertex_colors)[source_ids[existing]]):
        raise ValueError('Face step changed retained RGB')
    old_to_new = np.full(len(old), -1, int)
    old_to_new[source_ids[existing]] = np.flatnonzero(existing)
    updated = copy.deepcopy(data)
    names = list(data['parts'])
    labels = np.full(len(old), -1, int)
    for i, name in enumerate(names):
        ids = np.asarray(data['parts'][name]['vertex_indices'], int)
        if len(ids) and (ids.min() < 0 or ids.max() >= len(old)):
            raise ValueError('Input body-part index is out of bounds')
        if np.any(labels[ids] >= 0):
            raise ValueError('Input body-part labels overlap')
        labels[ids] = i
        if name not in FACE_PARTS and np.any(old_to_new[ids] < 0):
            raise ValueError('Face patch removed protected part: ' + name)
    remapped = np.full(len(xyz), names.index('face_neck'), int)
    remapped[existing] = labels[source_ids[existing]]
    for i, name in enumerate(names):
        part = updated['parts'][name]
        ids = np.flatnonzero(remapped == i)
        part.update(vertex_indices=ids.tolist(), vertex_count=len(ids), detected=bool(len(ids)),
                    center_mm=xyz[ids].mean(0).tolist() if len(ids) else None,
                    bbox_mm={'min': xyz[ids].min(0).tolist(), 'max': xyz[ids].max(0).tolist()} if len(ids) else None)
        if name == 'face_neck':
            part.update(label_origin='Retained Sapiens scan labels plus FLAME face patch assignment',
                        vote_agreement_median=None, minimum_supporting_views=0,
                        model_assigned_vertex_count=int((~existing).sum()))
    updated['unlabeled_vertex_count'] = int((remapped < 0).sum())
    return updated, old_to_new


def publish_face(path, report, data, original, candidate, source_ids, result):
    from run import publish_with_inherited_permissions
    from reconstruct_human import geometry_report

    if not result.get('geometry_checks_passed'):
        raise ValueError('Face candidate has not passed geometry checks')
    data, old_to_new = remap_parts(data, original, candidate, source_ids)
    xyz, faces = np.asarray(candidate.vertices), np.asarray(candidate.triangles)
    candidate.compute_vertex_normals()
    if not len(faces) or not np.isfinite(xyz).all() or not np.isfinite(np.asarray(candidate.vertex_normals)).all():
        raise ValueError('Invalid face mesh coordinates/normals')
    if faces.min() < 0 or faces.max() >= len(xyz):
        raise ValueError('Face connectivity is out of bounds')
    stem = Path(report['source']).stem
    final_ply = path.parent/(stem+'_mesh.ply')
    final_obj = final_ply.with_suffix('.obj')
    final_parts = path.parent/'body_parts.json'
    mapping_path = path.parent/'intermediate/flame/vertex_mapping.npz'
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    before = report['artifacts']['ply']['sha256']
    with tempfile.TemporaryDirectory(prefix='.flame_publish_', dir=path.parent) as temporary:
        scratch = Path(temporary)
        new_ply, new_obj, new_parts = scratch/final_ply.name, scratch/final_obj.name, scratch/final_parts.name
        if not o3d.io.write_triangle_mesh(str(new_ply), candidate, write_vertex_normals=True, write_vertex_colors=True):
            raise OSError('Face PLY export failed')
        loaded = o3d.io.read_triangle_mesh(str(new_ply))
        if not np.array_equal(np.asarray(loaded.vertices), xyz) or not np.array_equal(np.asarray(loaded.triangles), faces):
            raise ValueError('Face PLY readback differs from checked geometry')
        convert_ply_to_obj(new_ply, new_obj, vertex_colors=True)
        digest = file_hash(new_ply)
        data['mesh'].update(file=final_ply.name, sha256=digest, vertex_count=len(xyz), triangle_count=len(faces))
        data['fingerprint']['mesh_sha256'] = digest
        update = {'stage': FIELD, 'method': result['method'], 'source_mesh_sha256': before,
                  'output_mesh_sha256': digest, 'labels_transported_by_unchanged_vertex_indices': False,
                  'retained_vertices_mapped_by_explicit_indices': True,
                  'new_face_vertices_assigned_to_face_neck': True,
                  'vertex_mapping': str(mapping_path.relative_to(path.parent))}
        data['geometry_update'] = update
        data.setdefault('geometry_updates', []).append(update)
        data.setdefault('notes', []).append('FLAME changes face topology. Retained labels use explicit source indices; new face_neck labels are model assignments, not fresh image inference. Remap any external vertex-index landmarks using intermediate/flame/vertex_mapping.npz; removed face landmarks need manual review.')
        new_parts.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), 'utf-8')
        new_mapping = scratch/'vertex_mapping.npz'
        np.savez_compressed(new_mapping, source_vertex_index=source_ids, old_to_new=old_to_new,
                            source_mesh_sha256=np.array(before), output_mesh_sha256=np.array(digest))
        updated = copy.deepcopy(report)
        for key, final, staged in [('ply', final_ply, new_ply), ('obj', final_obj, new_obj)]:
            updated['artifacts'][key] = {'name': final.name, 'sha256': file_hash(staged)}
        updated['segmentation']['sha256'] = file_hash(new_parts)
        updated['segmentation']['labels_remapped_after_flame'] = True
        check = copy.deepcopy(candidate)
        check.scale(.001, center=np.zeros(3))
        updated['final_topology'] = geometry_report(check, check_intersections=False)
        updated['final_topology']['updated_after'] = FIELD
        centered = (xyz-candidate.get_center())*.001
        updated.setdefault('appearance', {})['exported_centered_signed_flux_m3'] = float(np.einsum(
            'ij,ij->i', centered[faces[:, 0]], np.cross(centered[faces[:, 1]], centered[faces[:, 2]])).sum()/6.)
        result = copy.deepcopy(result)
        result.update(source_mesh_sha256=before, output_mesh_sha256=digest,
                      vertex_mapping={'name': str(mapping_path.relative_to(path.parent)), 'sha256': file_hash(new_mapping)},
                      export_check={'ply_readback_matches': True, 'retained_coordinates_and_rgb_exact': True,
                                    'protected_part_membership_preserved': True, 'body_part_indices_remapped': True})
        updated[FIELD] = result
        # Texture packages refer to the earlier topology and must be rebuilt later.
        updated.pop('face_uv', None)
        files = [(new_obj, final_obj), (new_ply, final_ply), (new_parts, final_parts), (new_mapping, mapping_path)]
        centers_path = path.parent/'hand_centerlines.json'
        if centers_path.is_file():
            centers = json.loads(centers_path.read_text('utf-8'))
            centers['output_mesh_sha256'] = digest
            centers.setdefault('geometry_updates', []).append(update)
            new_centers = scratch/centers_path.name
            new_centers.write_text(json.dumps(centers, ensure_ascii=False, indent=2), 'utf-8')
            files.append((new_centers, centers_path))
            if 'hand_processing' in updated:
                updated['hand_processing']['centerlines'] = {'name': centers_path.name, 'sha256': file_hash(new_centers)}
        staged_report = scratch/'report.json'
        staged_report.write_text(json.dumps(updated, ensure_ascii=False, indent=2), 'utf-8')
        files.append((staged_report, path))
        existed = [destination.exists() for _, destination in files]
        for i, (_, destination) in enumerate(files):
            if existed[i]:
                shutil.copyfile(destination, scratch/f'backup_{i}')
        try:
            for staged, destination in files:
                publish_with_inherited_permissions(staged, destination)
        except Exception:
            for i, (_, destination) in enumerate(files):
                if existed[i]:
                    publish_with_inherited_permissions(scratch/f'backup_{i}', destination)
                else:
                    destination.unlink(missing_ok=True)
            raise
    return updated


def apply_face_refinement(report_path, enabled=True, axis='auto', up_sign=1):
    path = Path(report_path)
    report = json.loads(path.read_text('utf-8'))
    if report.get(FIELD, {}).get('status') == 'complete':
        print('FLAME face: reusing completed result (no repeated modification).', flush=True)
        return report
    reason = None
    if not enabled:
        status, reason = 'disabled', 'Disabled by processing options'
    elif report.get('subject', {}).get('kind') != 'human' or report.get('segmentation', {}).get('status') != 'complete':
        status, reason = 'skipped', 'Requires verified human body-part positions'
    if reason:
        report[FIELD] = {'status': status, 'applied': False, 'reason': reason}
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
        return report
    start = time.perf_counter()
    try:
        print('\n[FLAME: fit face / preserve seam / remap labels]', flush=True)
        source = path.parent/report['artifacts']['ply']['name']
        data = json.loads((path.parent/'body_parts.json').read_text('utf-8'))
        digest = file_hash(source)
        if digest != data['mesh']['sha256'] or digest != report['artifacts']['ply']['sha256']:
            raise ValueError('FLAME input labels, mesh and report hashes disagree')
        original = o3d.io.read_triangle_mesh(str(source))
        original.compute_vertex_normals()
        up = int(np.argmax(np.ptp(np.asarray(original.vertices), axis=0))) if axis == 'auto' else 'xyz'.index(axis)
        if up != 0 or up_sign != 1:
            raise ValueError('FLAME currently requires capture coordinates X-up / -Z-forward')
        for name in ['face_neck', 'hair']:
            if len(data['parts'].get(name, {}).get('vertex_indices', [])) < 50:
                raise ValueError('Insufficient face/hair semantic support')
        template, basis, faces, model_digest = load_model()
        fitted, nose, fit_detail = fit_template(original, data['parts'], template, basis, faces)
        fit_quality = fit_detail['history'][-1]
        if fit_quality['median_correspondence_mm'] > 5 or fit_quality['p90_correspondence_mm'] > 10:
            raise ValueError('Face template fitting residuals exceed limits')
        fitted_mesh = make_mesh(fitted, faces)
        out = path.parent/'intermediate/flame'
        out.mkdir(parents=True, exist_ok=True)
        fit_path = out/'fit.json'
        fit_path.write_text(json.dumps({'model_sha256': model_digest, 'source_mesh_sha256': digest,
                                        'fit': fit_detail}, ensure_ascii=False, indent=2), 'utf-8')
        attempts = []
        selected = None
        for style in ['regional', 'uniform']:
            for radius in [1., .9, .8]:
                try:
                    candidate, mapping, checks = patch(original, fitted_mesh, data['parts'], nose, radius_scale=radius,
                                                       style=style, anchors=fit_detail.get('region_anchors'))
                    # Validate protected semantic membership before accepting the candidate.
                    if checks['geometry_checks_passed']:
                        remap_parts(data, original, candidate, mapping)
                        selected = (candidate, mapping, checks)
                    attempts.append(checks)
                    if selected is not None:
                        break
                except (ValueError, RuntimeError) as error:
                    attempts.append({'radius_scale': radius, 'blend_style': style,
                                     'geometry_checks_passed': False, 'reason': str(error)})
            if selected is not None:
                break
        result = {'status': 'complete', 'applied': selected is not None, 'method': 'FLAME fitted face patch with fixed scan seam',
                  'model_sha256': model_digest, 'fit': str(fit_path.relative_to(path.parent)),
                  'fit_quality': fit_quality, 'attempts': attempts,
                  'model_imputed_geometry_not_measurement_truth': True,
                  'elapsed_seconds': round(time.perf_counter()-start, 3)}
        if selected:
            candidate, mapping, checks = selected
            result.update(geometry_checks_passed=True, selected=checks)
            report = publish_face(path, report, data, original, candidate, mapping, result)
        else:
            result.update(reason='No patch passed geometry and protected-region checks', previous_geometry_retained=True)
            report[FIELD] = result
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
        print('FLAME face:', 'applied' if selected else 'previous geometry retained', flush=True)
    except Exception as error:
        report[FIELD] = {'status': 'failed', 'applied': False, 'reason': str(error),
                         'previous_geometry_retained': True, 'elapsed_seconds': round(time.perf_counter()-start, 3)}
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
        print('FLAME face kept previous geometry:', str(error), flush=True)
    return report
