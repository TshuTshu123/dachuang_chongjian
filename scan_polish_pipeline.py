"""正式流程的 MANO 表面融合、腕部校正及躯干去噪；逐步检查并原子发布。

每一步失败保留该步输入，已通过的前一步仍然有效。部位标签随细分更新，
坐标、包围盒、PLY/OBJ/部位哈希及中心线关联一起刷新，不依赖 experiments。
"""
import copy
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from body_parts import file_hash
from ply_to_obj import convert_ply_to_obj
from scan_polish import refine_mano_surface, refine_wrists, refine_torso, QualityRejection


def rejection_key(report, raw_unit, axis, up_sign):
    """Only deterministic refusals are memoized; I/O/publication errors retry."""
    root = Path(__file__).resolve().parent
    return {'source_sha256': file_hash(report['source']), 'axis': axis, 'up_sign': up_sign,
            'raw_unit': raw_unit, 'pipeline_fingerprint': report.get('fingerprint', {}),
            'code': {name: file_hash(root/name) for name in
                     ('scan_polish.py', 'scan_polish_pipeline.py', 'tools/torso_surface_trial.py')}}


def publish_step(path, report, data, original, candidate, parts, result, field):
    from run import publish_with_inherited_permissions

    xyz = np.asarray(candidate.vertices)
    faces = np.asarray(candidate.triangles)
    old = np.asarray(original.vertices)
    source_hash = report['artifacts']['ply']['sha256']
    if not len(faces) or not np.isfinite(xyz).all():
        raise ValueError('Nonfinite or empty polished mesh')
    candidate.compute_vertex_normals()
    if not np.isfinite(np.asarray(candidate.vertex_normals)).all():
        raise ValueError('Nonfinite polished mesh normals')
    if not result.get('topology_changed'):
        if not np.array_equal(faces, np.asarray(original.triangles)) or len(xyz) != len(old):
            raise ValueError('Unexpected change to face connectivity')
        if not np.array_equal(np.asarray(candidate.vertex_colors), np.asarray(original.vertex_colors)):
            raise ValueError('Unexpected color change')
    else:
        if len(xyz) < len(old) or not np.array_equal(np.asarray(candidate.vertex_colors)[:len(old)], np.asarray(original.vertex_colors)):
            raise ValueError('Subdivision did not retain original vertex colors/order')
    if candidate.is_vertex_manifold() != original.is_vertex_manifold():
        raise ValueError('Vertex manifold check regressed')
    if len(candidate.get_non_manifold_edges()) != len(original.get_non_manifold_edges()):
        raise ValueError('Edge manifold check regressed')
    data = copy.deepcopy(data)
    data['parts'] = copy.deepcopy(parts)
    assigned = np.zeros(len(xyz), bool)
    for part in data['parts'].values():
        ids = np.asarray(part['vertex_indices'], dtype=int)
        if len(ids) and (ids.min() < 0 or ids.max() >= len(xyz) or len(np.unique(ids)) != len(ids)):
            raise ValueError('Invalid transported body-part indices')
        assigned[ids] = True
        part.update(vertex_count=len(ids), detected=bool(len(ids)),
                    center_mm=xyz[ids].mean(0).tolist() if len(ids) else None,
                    bbox_mm={'min': xyz[ids].min(0).tolist(), 'max': xyz[ids].max(0).tolist()} if len(ids) else None)
    # Protected regions are an independent postcondition, in addition to per-method checks.
    protected = ('face_neck', 'hair', 'left_shoe', 'right_shoe', 'lower_clothing')
    if field == 'torso_surface_refinement':
        protected += ('left_hand', 'right_hand', 'left_lower_arm', 'right_lower_arm')
    for name in protected:
        ids = np.asarray(parts.get(name, {}).get('vertex_indices', []), dtype=int)
        ids = ids[ids < len(old)]
        if not np.array_equal(xyz[ids], old[ids]):
            raise ValueError(f'Protected region moved: {name}')
    stem = Path(report['source']).stem
    final_ply = path.parent / (stem + '_mesh.ply')
    final_obj = final_ply.with_suffix('.obj')
    final_parts = path.parent / 'body_parts.json'
    centers_path = path.parent / 'hand_centerlines.json'
    with tempfile.TemporaryDirectory(prefix='.scan_polish_', dir=path.parent) as scratch:
        scratch = Path(scratch)
        new_ply, new_obj, new_parts = scratch/final_ply.name, scratch/final_obj.name, scratch/final_parts.name
        if not o3d.io.write_triangle_mesh(str(new_ply), candidate, write_vertex_normals=True, write_vertex_colors=True):
            raise OSError('Could not export polished PLY')
        read = o3d.io.read_triangle_mesh(str(new_ply))
        if not np.array_equal(np.asarray(read.vertices), xyz) or not np.array_equal(np.asarray(read.triangles), faces):
            raise ValueError('Exported PLY differs from checked candidate')
        convert_ply_to_obj(new_ply, new_obj, vertex_colors=True)
        digest = file_hash(new_ply)
        data['mesh'].update(file=final_ply.name, sha256=digest, vertex_count=len(xyz), triangle_count=len(faces))
        data['fingerprint']['mesh_sha256'] = digest
        data['unlabeled_vertex_count'] = int((~assigned).sum())
        update = {'method': result['method'], 'stage': field, 'source_mesh_sha256': source_hash,
                  'labels_transported_by_unchanged_vertex_indices': True,
                  'new_hand_vertices_assigned_from_subdivision': bool(result.get('topology_changed'))}
        data['geometry_update'] = update
        data.setdefault('geometry_updates', []).append(update)
        data.setdefault('notes', []).append(f'{field}: original vertex indices retained; subdivided hand vertices inherit hand assignment; centers, bounds and hashes refreshed.')
        new_parts.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        updated = copy.deepcopy(report)
        for key, name, staged in (('ply', final_ply, new_ply), ('obj', final_obj, new_obj)):
            updated['artifacts'][key] = {'name': name.name, 'sha256': file_hash(staged)}
        updated['segmentation']['sha256'] = file_hash(new_parts)
        updated['segmentation']['labels_transported_after_' + field] = True
        result.update(source_mesh_sha256=source_hash, output_mesh_sha256=digest,
                      export_check={'finite_vertices_and_normals': True, 'ply_readback_matches': True,
                                    'protected_regions_unchanged': True, 'manifold_checks_match_input': True,
                                    'body_part_indices_valid': True})
        files = [(new_obj, final_obj), (new_ply, final_ply), (new_parts, final_parts)]
        if centers_path.is_file():
            centers = json.loads(centers_path.read_text(encoding='utf-8'))
            centers['output_mesh_sha256'] = digest
            centers.setdefault('geometry_updates', []).append({'stage': field,
                'source_mesh_sha256': source_hash, 'output_mesh_sha256': digest})
            centers['geometry_note'] = 'Centerlines estimated before template/surface updates; not manual measurement landmarks or recomputed joints.'
            new_centers = scratch / centers_path.name
            new_centers.write_text(json.dumps(centers, ensure_ascii=False, indent=2), encoding='utf-8')
            files.append((new_centers, centers_path))
            if 'hand_processing' in updated:
                updated['hand_processing']['centerlines'] = {'name': centers_path.name, 'sha256': file_hash(new_centers)}
        centered = (xyz - candidate.get_center()) * .001
        updated.setdefault('appearance', {})['exported_centered_signed_flux_m3'] = float(np.einsum(
            'ij,ij->i', centered[faces[:, 0]], np.cross(centered[faces[:, 1]], centered[faces[:, 2]])).sum()/6)
        topology = copy.deepcopy(updated.get('final_topology', {}))
        topology.update(vertices=len(xyz), triangles=len(faces), updated_after=field)
        if result.get('topology_changed'):
            from reconstruct_human import geometry_report
            check = copy.deepcopy(candidate)
            check.scale(.001, center=np.zeros(3))
            topology = geometry_report(check, check_intersections=False)
            topology['updated_after'] = field
        updated['final_topology'] = topology
        updated[field] = result
        staged_report = scratch / 'report.json'
        staged_report.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding='utf-8')
        files.append((staged_report, path))
        existed = []
        for index, (_, destination) in enumerate(files):
            existed.append(destination.exists())
            if existed[-1]:
                shutil.copyfile(destination, scratch / f'backup_{index}')
        try:
            for staged, destination in files:
                publish_with_inherited_permissions(staged, destination)
        except Exception:
            for index, (_, destination) in enumerate(files):
                if existed[index]:
                    publish_with_inherited_permissions(scratch / f'backup_{index}', destination)
                else:
                    destination.unlink(missing_ok=True)
            raise
    return updated


def apply_scan_polish(report_path, hand_enabled=True, torso_enabled=True, raw_unit='mm', axis='auto', up_sign=1):
    path = Path(report_path)
    report = json.loads(path.read_text(encoding='utf-8'))
    raw = None
    stages = (('mano_surface_refinement', hand_enabled), ('wrist_alignment', hand_enabled),
              ('torso_surface_refinement', torso_enabled))
    for field, enabled in stages:
        previous = report.get(field, {})
        reusable = previous.get('status') == 'complete'
        if reusable and previous.get('rejection_kind') == 'quality':
            reusable = previous.get('rejection_key') == rejection_key(report, raw_unit, axis, up_sign)
        if reusable:
            print(field + ': reusing completed result (no repeated modification).', flush=True)
            continue
        reason = None
        if not enabled:
            status, reason = 'disabled', 'Disabled by processing options'
        elif report.get('subject', {}).get('kind') != 'human' or report.get('segmentation', {}).get('status') != 'complete':
            status, reason = 'skipped', 'Requires human body-part positions'
        elif field != 'torso_surface_refinement' and not report.get('mano_hand_processing', {}).get('applied'):
            status, reason = 'skipped', 'MANO replacement was not adopted; retain original hands'
        if reason:
            report[field] = {'status': status, 'applied': False, 'reason': reason}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            continue
        started = time.perf_counter()
        try:
            print('\n[' + field + ': checked candidate]', flush=True)
            source = path.parent / report['artifacts']['ply']['name']
            data = json.loads((path.parent/'body_parts.json').read_text(encoding='utf-8'))
            digest = file_hash(source)
            if digest != data['mesh']['sha256'] or digest != report['artifacts']['ply']['sha256']:
                raise ValueError('Polish input mesh, report and body-part hashes disagree')
            original = o3d.io.read_triangle_mesh(str(source))
            original.compute_vertex_normals()
            if raw is None:
                raw = np.asarray(o3d.io.read_point_cloud(report['source']).points).copy()
                raw = raw[np.isfinite(raw).all(1)] * (1000 if raw_unit == 'm' else 1)
            if field == 'torso_surface_refinement':
                candidate, result, parts = refine_torso(original, raw, data['parts'], axis, up_sign)
            else:
                fit = json.loads((path.parent/'intermediate/mano/report.json').read_text(encoding='utf-8'))
                if field == 'mano_surface_refinement':
                    palms = {p['part']: p for p in report['hand_processing']['palm_volume']['parts']}
                    candidate, result, parts = refine_mano_surface(original, raw, data['parts'], fit, palms)
                else:
                    scanmesh = o3d.io.read_triangle_mesh(str(path.parent/report['artifacts']['base']['name']))
                    candidate, result, parts = refine_wrists(original, raw, data['parts'], fit, scanmesh)
            result['elapsed_seconds'] = round(time.perf_counter()-started, 3)
            if field == 'torso_surface_refinement' and not result['applied']:
                result.update(rejection_kind='quality',
                              rejection_key=rejection_key(report, raw_unit, axis, up_sign))
            if result['applied']:
                report = publish_step(path, report, data, original, candidate, parts, result, field)
            else:
                report[field] = result
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(field, 'applied' if result['applied'] else 'previous geometry retained', flush=True)
        except QualityRejection as error:
            report[field] = {'status': 'complete', 'applied': False, 'reason': str(error),
                             'rejection_kind': 'quality', 'previous_geometry_retained': True,
                             'rejection_key': rejection_key(report, raw_unit, axis, up_sign),
                             'elapsed_seconds': round(time.perf_counter()-started, 3)}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(field, 'checked rejection retained:', error, flush=True)
        except Exception as error:
            report[field] = {'status': 'failed', 'applied': False, 'reason': str(error),
                             'previous_geometry_retained': True, 'elapsed_seconds': round(time.perf_counter()-started, 3)}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(field, 'kept previous geometry:', error, flush=True)
    return report
