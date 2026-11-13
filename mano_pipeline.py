"""正式 MANO 手部步骤：由 run.py 自动调用，拼接失败时保留原网格。

python run.py --all：处理所有输入；--no-hand-refine 同时关闭 MANO。
模型在 tools/mano，局部预览与参数在本次输出 intermediate/mano。
新模板与腕口改变顶点编号，部位索引按保留坐标及新增手部重新映射。
"""
import copy
import json
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from body_parts import file_hash
from ply_to_obj import convert_ply_to_obj
from tools.mano_fit import trial
from tools.mano_join import join
from hand_thickness_policy import compare_pair


def apply_mano(report_path, enabled=True):
    from run import publish_with_inherited_permissions
    path = Path(report_path); report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('mano_hand_processing', {}).get('status') == 'complete':return report
    if not enabled or report.get('subject', {}).get('kind') != 'human':
        report['mano_hand_processing'] = {'status': 'disabled' if not enabled else 'skipped', 'applied': False, 'reason': 'Requires enabled human hand processing'}
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8');return report
    started = time.perf_counter(); out = path.parent/'intermediate/mano'
    try:
        print('\n[MANO: fit templates and join checked wrist seams]', flush=True)
        positions = path.parent/'body_parts.json'; data = json.loads(positions.read_text(encoding='utf-8'))
        source = path.parent/report['artifacts']['ply']['name']; old_hash = file_hash(source)
        if data['mesh']['sha256'] != old_hash:raise ValueError('MANO positions do not match source mesh')
        if not (path.parent/'hand_centerlines.json').is_file():raise ValueError('Reliable hand centerlines unavailable')
        if any(not data['parts'].get(name, {}).get('detected') for name in ['left_hand','right_hand']):raise ValueError('Both hand regions required')
        original = o3d.io.read_triangle_mesh(str(source)); old = np.asarray(original.vertices)
        trial(0, report_path=path, output_folder=out)
        selected_hands, thickness_policy = compare_pair(report, original, data, out)
        if not selected_hands:
            report['mano_hand_processing']={'status':'complete','applied':False,'reason':'Pair comparison does not justify replacing original hands','thickness_policy':thickness_policy,'details':'intermediate/mano/report.json'}
            path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report
        attempts = []; result = None
        for cut_offset in [0.,2.,5.,8.]:
            sides=selected_hands
            try:
                candidate_result = join(0, output_folder=out, part_names=sides, smooth_palm_wrist=True, wrist_cut_offset=cut_offset)
                attempts.append(candidate_result['join_attempt']); result = candidate_result
                if result['wrist_stitched']:break
            except (ValueError, RuntimeError) as error:
                attempts.append({'parts': sides, 'wrist_cut_offset_mm':cut_offset,'reason': str(error)})
        if result is None:
            result = {'wrist_stitched': False, 'join_attempt': attempts[-1]}
        result['all_join_attempts'] = attempts
        if not result['wrist_stitched']:
            report['mano_hand_processing'] = {'status': 'complete', 'applied': False, 'reason': 'Join rejected; original hand retained', 'join_attempt': result['join_attempt'], 'all_join_attempts': attempts}
        else:
            candidate_path = out/result['full_preview']; candidate = o3d.io.read_triangle_mesh(str(candidate_path)); xyz = np.asarray(candidate.vertices)
            distance, nearest = cKDTree(old).query(xyz)
            names = list(data['parts']); labels = np.full(len(old), -1, int)
            for i, name in enumerate(names):labels[data['parts'][name]['vertex_indices']] = i
            remapped = labels[nearest]
            new = distance > 1.e-7
            wrists = np.array([p['wrist_mm'] for p in result['parts']])
            hands = [p['part'] for p in result['parts']]
            owner = cKDTree(wrists).query(xyz[new])[1]
            remapped[new] = [names.index(hands[k]) for k in owner]
            for i, name in enumerate(names):
                part = data['parts'][name]; ids = np.flatnonzero(remapped == i)
                part.update(vertex_indices=ids.tolist(), vertex_count=len(ids), detected=bool(len(ids)))
                part['center_mm'] = xyz[ids].mean(0).tolist() if len(ids) else None
                part['bbox_mm'] = {'min': xyz[ids].min(0).tolist(), 'max': xyz[ids].max(0).tolist()} if len(ids) else None
                if name in [p['part'] for p in result['join_attempt']['parts']]:
                    part.update(vote_agreement_median=None, minimum_supporting_views=0, label_origin='MANO template assignment, not image votes')
            from reconstruct_human import geometry_report
            qa = copy.deepcopy(candidate);qa.scale(.001, center=np.zeros(3))
            topology = geometry_report(qa, check_intersections=False)
            topology['updated_after'] = 'MANO'
            stem = Path(report['source']).stem; final_mesh = path.parent/(stem+'_mesh.ply'); final_obj = final_mesh.with_suffix('.obj')
            with tempfile.TemporaryDirectory(prefix='.mano_publish_', dir=path.parent) as scratch:
                scratch = Path(scratch); new_mesh = scratch/final_mesh.name; new_obj = scratch/final_obj.name; new_positions = scratch/'body_parts.json'
                shutil.copyfile(candidate_path, new_mesh); convert_ply_to_obj(new_mesh, new_obj, vertex_colors=True)
                digest = file_hash(new_mesh)
                data['mesh'].update(file=final_mesh.name, sha256=digest, vertex_count=len(xyz), triangle_count=len(candidate.triangles))
                data['fingerprint']['mesh_sha256'] = digest
                data['unlabeled_vertex_count'] = int((remapped < 0).sum())
                data['geometry_update'] = {'method': 'MANO wrist-joined templates', 'source_mesh_sha256': old_hash, 'labels_transported_by_unchanged_vertex_indices': False, 'retained_vertices_matched_by_exact_coordinates': True, 'new_hand_vertices_assigned_from_join_regions': True}
                data['notes'].append('MANO changes mesh indices; retained body labels mapped by coordinates, new hand labels are model assignments, not a second image inference.')
                new_positions.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
                files = [(new_obj, final_obj), (new_mesh, final_mesh), (new_positions, positions)]
                existed = []
                for index, (_, destination) in enumerate(files):
                    existed.append(destination.exists())
                    if existed[-1]:shutil.copyfile(destination, scratch/f'backup_{index}')
                try:
                    for staged, destination in files:publish_with_inherited_permissions(staged, destination)
                except Exception:
                    for index, (_, destination) in enumerate(files):
                        if existed[index]:publish_with_inherited_permissions(scratch/f'backup_{index}', destination)
                        else:destination.unlink(missing_ok=True)
                    raise
            for key, dest in [('ply', final_mesh), ('obj', final_obj)]:report['artifacts'][key] = {'name': dest.name, 'sha256': file_hash(dest)}
            report['segmentation']['sha256'] = file_hash(positions)
            report['final_topology'] = topology
            centered = (xyz-candidate.get_center())*.001; f = np.asarray(candidate.triangles)
            report['appearance']['exported_centered_signed_flux_m3'] = float(np.einsum('ij,ij->i', centered[f[:,0]], np.cross(centered[f[:,1]],centered[f[:,2]])).sum()/6)
            centers_path = path.parent/'hand_centerlines.json'; centers = json.loads(centers_path.read_text(encoding='utf-8'))
            centers['output_mesh_sha256'] = digest;centers['geometry_note'] = 'Paths estimated before MANO replacement; use intermediate/mano/report.json joints for fitted hands.'
            centers_path.write_text(json.dumps(centers, ensure_ascii=False, indent=2), encoding='utf-8')
            if 'hand_processing' in report:report['hand_processing']['centerlines'] = {'name': centers_path.name, 'sha256': file_hash(centers_path)}
            report['mano_hand_processing'] = {'status': 'complete', 'applied': True, 'method': result['method'], 'join_attempt': result['join_attempt'], 'details': 'intermediate/mano/report.json', 'source_mesh_sha256': old_hash, 'output_mesh_sha256': digest}
            report['mano_hand_processing']['all_join_attempts'] = result.get('all_join_attempts', [result['join_attempt']])
            result['full_preview'] = '../../'+final_mesh.name
            (out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            candidate_path.unlink()
        report['mano_hand_processing']['elapsed_seconds'] = round(time.perf_counter()-started, 2)
        report['mano_hand_processing']['thickness_policy'] = thickness_policy
    except Exception as error:
        report['mano_hand_processing'] = {'status': 'failed', 'applied': False, 'reason': str(error), 'previous_geometry_retained': True}
        print('MANO kept previous hands:', str(error), flush=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
