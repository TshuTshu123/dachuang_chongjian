"""日常入口：自动管理输入输出，以已验证的 scan 参数重建并导出 PLY/OBJ。
  python run.py              处理 data 中最新修改的 PLY
  python run.py 3            处理 export_00003.ply，也可直接写 PLY 路径
  --all：处理全部数据；--redo：忽略缓存重建；--fast：粗略快速预览
  --kind auto：自动判断人体，否则走通用物体；human/object 可手动指定。
  默认尝试通过质量筛选的面部/身体轻度表面整理，手部保持固定；--no-surface-refine 关闭。
  --unit m：输入为米（默认 mm）；--sole-axis x：显式开启鞋底约束并指定竖直轴
  --up-sign -1：该轴负向朝上；--sole-axis off：关闭鞋底外扩约束
  默认关闭鞋底删面及额外双边过滤；--refine 可显式开启额外过滤。
  默认尝试鞋底截断补面（新数据为测试功能）；--no-shoe-repair 可关闭。
  --shoe-axis x/y/z：手动指定鞋底竖直轴，默认 auto 最长轴，--up-sign -1 为负向朝上。
  少量非流形边导致朝向失败时自动限量清理；失败原因写入 failure.json。
  默认自动修正法线并修补孤立黑点；--normals-only 只修法线，--no-appearance 全部关闭。
  data_raw/编号 下有配套 RAW 时优先用多视角补色；--no-raw-color 使用单 PLY 的黑点修补。
  配套照片和脸颈标签通过检查时额外生成 textured/face_uv.obj；--no-face-uv 跳过。
  默认在 Meta/Sapiens 识别后自动检查手部密度、厚度与观测贴合，收益不足则保留原形。
  掌部以腕部/椭圆截面参考整体收紧，指尖和手腕处渐变；不需要单独选择掌部参数。
  默认比较双手 MANO 候选；明显过厚且候选改善时成对替换，腕口以真实边界/切向连接掌侧。
  MANO 采用后继续融合掌部观测并以原扫描腕部校正衔接；身体表面增加局部二次曲面去噪。
  默认用 FLAME 拟合并局部重建正脸，固定边界并重映射部位索引；失败保留原脸。
  --no-face-refine 关闭 FLAME；--fast、--no-surface-refine 同时跳过该步骤。
  settings.json 分别控制 MANO/FLAME 套用；--hand-model/--no-hand-model、--face-model/--no-face-model 可临时覆盖。
  --no-hand-refine：关闭自动手部处理与 MANO；--no-segmentation 同时跳过识别和手部处理。
  --hand-refine：兼容旧的掌面实验；日常无需选择，--no-hand-refine 关闭自动手部处理。
  --hand-skeleton：试用原有稳定中心线限定的手指表面贴合；默认关闭。
  --hand-density：额外试验低密度虚接过滤和观测厚度约束，可能留下开口；默认关闭。
  --seg-python 路径：GPU Python（默认本机 test0）；--seg-axis x/y/z：指定识别竖直轴。
输出：outputs/models/输入名/时间/，中间结果在 intermediate，旧结果不覆盖。
无需运行其它脚本；自动参数不等于保证恢复扫描缺失的真实细节。
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import shutil
import tempfile
import uuid
import time
from datetime import datetime

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from mesh_refine import bilateral_mesh_filter, observation_support
from ply_to_obj import convert_ply_to_obj
from reconstruct_human import geometry_report, transfer_colors
from sole_repair import shoe_plan, apply_shoe_plan
from mesh_appearance import apply_appearance
from body_parts import ensure_body_parts, DEFAULT_GPU_PYTHON
from palm_pipeline import refine_hands
from scene_policy import classify_parts, generic_observation_trim, repair_generic_topology
from surface_evaluation import refine_surface
from tools.raw_capture import raw_color_input, apply_raw_colors
from tools.face_uv import export_face_uv
from skeleton_hand import refine_skeleton_hands
from mano_pipeline import apply_mano
from scan_polish_pipeline import apply_scan_polish
from face_pipeline import apply_face_refinement
from settings import resolve_model_settings

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW_DATA = ROOT / 'data_raw'
OUTPUTS = ROOT / 'outputs' / 'models'
WORKERS = min(8, os.cpu_count() or 1)
VERSION = 28


def select_inputs(value=None, all_files=False, data_dir=DATA):
    data_dir = Path(data_dir)
    available = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == '.ply') if data_dir.is_dir() else []
    if value is not None:
        if value.isdecimal():
            matches = [p for p in available
                       if (match := re.fullmatch(r'export_?(\d+)', p.stem, flags=re.IGNORECASE))
                       and int(match.group(1)) == int(value)]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one export PLY for number {value}, found {len(matches)}.')
            return matches
        path = Path(value)
        if not path.is_file():
            path = data_dir / value
        if not path.is_file() or path.suffix.lower() != '.ply':
            raise ValueError(f'PLY not found: {value}')
        return [path.resolve()]
    if not available:
        raise ValueError(f'No PLY files in {data_dir}')
    return available if all_files else [max(available, key=lambda p: (p.stat().st_mtime_ns, p.name))]


def choose_parameters(points_mm, fast=False):
    """Bounded heuristics for human scans, not a ground-truth optimizer.

    Use full-cloud neighborhoods around sampled queries, never a thinning of
    the samples themselves. Normal spread also contains real curvature.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError('At least 100 finite XYZ points are required.')
    robust_extent = np.quantile(points, .995, axis=0) - np.quantile(points, .005, axis=0)
    body_size = float(robust_extent.max())
    if not 200 <= body_size <= 5000:
        raise ValueError('Input is outside expected human-scan size (0.2..5 m). Check --unit mm or --unit m.')
    tree = cKDTree(points)
    sample = points[np.random.default_rng(0).choice(len(points), min(6000, len(points)), replace=False)]
    distances, indices = tree.query(sample, k=32, workers=WORKERS)
    spacing = float(np.median(distances[:, 1]))
    if spacing <= 0:
        raise ValueError('Point spacing is zero; remove duplicate/invalid observations.')
    delta = points[indices] - sample[:, None, :]
    centered = delta - delta.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / centered.shape[1]
    eigenvalues, axes = np.linalg.eigh(covariance)
    residual = np.einsum('nkj,nj->nk', centered, axes[:, :, 0])
    spread = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=1, keepdims=True)), axis=1)
    usable = ((eigenvalues[:, 1] > 1e-8) & (distances[:, -1] < 30)
              & (eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], 1e-30) < .2))
    noise = float(np.median(spread[usable])) if usable.any() else float(np.median(spread))
    # Regression on export_00001/00003 reproduced history with these settings.
    # Neighborhood/radius changes can change the normal graph and winding,
    # so raw spread alone must not select a new fine reconstruction policy.
    sparse = spacing > 4
    voxel_mm = 3.0 if fast or sparse else 2.0
    normal_mm = 16.0 if sparse else 12.0
    depth = 8 if fast else 9
    return {
        'sampled_spacing_mm': spacing, 'local_spread_proxy_mm': noise,
        'local_spread_note': 'Includes curvature; not calibrated sensor noise or accuracy.',
        'robust_extent_mm': robust_extent.tolist(), 'sampled_queries': len(sample),
        'voxel_mm': round(voxel_mm, 3), 'depth': depth,
        'normal_radius_mm': round(normal_mm, 3), 'normal_neighbors': 64,
        'point_iterations': 1 if fast else 2,
        'point_move_cap_mm': 1., 'cluster_radius_mm': 12., 'smooth_iterations': 2,
        'parameter_policy': 'validated_scan_baseline' if not sparse else 'sparse_conservative_fallback',
        'policy_note': 'Statistics are diagnostic, not an accuracy objective. Sparse fallback is not regression-validated.',
        'mesh_move_cap_mm': round(float(np.clip(.6 * noise, .15, .75)), 3),
        'fast': fast,
    }


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_with_inherited_permissions(source, destination):
    """Atomically publish through an ordinary sibling, inheriting output ACLs.
    Never rename directly out of TemporaryDirectory: Windows 0700 temp ACLs
    would follow the file and block the other account from opening/deleting it.
    """
    source=Path(source);destination=Path(destination)
    staging=destination.with_name(f'.{destination.stem}.{uuid.uuid4().hex}.publish{destination.suffix}')
    try:
        with source.open('rb') as reader,staging.open('xb') as writer:
            shutil.copyfileobj(reader,writer,1024*1024)
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)


def fingerprint(source, unit, fast):
    code = {name: file_hash(ROOT / name) for name in
            ('run.py', 'reconstruct_human.py', 'surface_detail.py', 'mesh_refine.py', 'ply_to_obj.py', 'sole_repair.py', 'mesh_appearance.py',
             'palm_pipeline.py', 'hand_pipeline.py', 'finger_restore.py', 'scene_policy.py', 'surface_quality.py', 'surface_evaluation.py', 'body_parts.py', 'tools/sapiens/infer.py', 'tools/sapiens/render_views.py',
             'tools/sapiens/checkpoint_manifest.json', 'tools/sapiens/vendor/classes_and_palettes.py',
             'tools/raw_capture.py', 'tools/face_uv.py', 'hand_centerlines.py', 'skeleton_hand.py', 'hand_density_constraints.py', 'palm_volume.py', 'mano_pipeline.py', 'tools/mano_fit.py', 'tools/mano_join.py', 'finger_envelope.py', 'hand_thickness_policy.py', 'tools/wrist_cut_trial.py',
             'scan_polish.py', 'scan_polish_pipeline.py', 'tools/mano_surface_trial.py', 'tools/wrist_observation_trial.py', 'tools/torso_surface_trial.py', 'tools/surface_trials.py',
             'face_refine.py', 'face_pipeline.py', 'requirements.txt', 'settings.py')}
    return {'version': VERSION, 'source_sha256': file_hash(source), 'unit': unit,
            'fast': fast, 'code_sha256': code, 'open3d_version': o3d.__version__,
            'numpy_version': np.__version__, 'mano_assets': {name: file_hash(ROOT/'tools/mano'/name) if (ROOT/'tools/mano'/name).is_file() else None for name in ('MANO_LEFT.pkl','MANO_RIGHT.pkl')},
            'flame_assets': {name: file_hash(ROOT/'tools/flame'/name) if (ROOT/'tools/flame'/name).is_file() else None for name in ('flame2023_Open.pkl', 'download_manifest.json')}}


def cached_result(folder, signature):
    for report_path in sorted(folder.glob('*/report.json'), reverse=True):
        try:
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if report.get('fingerprint') != signature:
                continue
            artifacts = report.get('artifacts', {})
            if set(artifacts) != {'ply', 'obj', 'base'}:
                continue
            valid = True
            for artifact in artifacts.values():
                path = report_path.parent / artifact['name']
                if not path.resolve().is_relative_to(report_path.parent.resolve()) or not path.is_file() or file_hash(path) != artifact['sha256']:
                    valid = False
                    break
            face = report.get('flame_face_refinement', {})
            if valid and face.get('applied'):
                mapping = face.get('vertex_mapping', {})
                mapping_path = report_path.parent/mapping.get('name', '')
                if (not mapping_path.resolve().is_relative_to(report_path.parent.resolve())
                        or not mapping_path.is_file() or file_hash(mapping_path) != mapping.get('sha256')):
                    valid = False
            if valid:
                return report_path, report
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def add_body_part_positions(report_path, enabled=True, gpu_python=None, axis='auto', up_sign=1):
    """Optional sidecar has its own cache; it must not change/rebuild geometry."""
    report_path=Path(report_path)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('subject',{}).get('kind') in ('object','unknown'):
        source=report_path.parent/report['artifacts']['ply']['name']
        sidecar=report_path.parent/'body_parts.json'
        data={'schema_version':1,'status':'not_detected' if report['subject']['kind']=='object' else 'unavailable',
              'mesh':{'file':source.name,'sha256':file_hash(source)},'subject':report['subject'],'parts':{},
              'reason':'Generic object branch; human position labels are not assigned.'}
        sidecar.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        report['segmentation']={'status':data['status'],'path':str(sidecar),'sha256':file_hash(sidecar)}
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return report
    if enabled:
        source=report_path.parent/report['artifacts']['ply']['name']
        started=time.perf_counter()
        try:
            result=ensure_body_parts(source,gpu_python=gpu_python,axis=axis,up_sign=up_sign,
                                     output=report_path.parent/'body_parts.json')
        except Exception as error:
            # Segmentation is an auxiliary feature. Keep a successfully exported
            # reconstruction even if its environment or optional code fails.
            reason=f'{type(error).__name__}: {error}'
            sidecar=report_path.parent/'body_parts.json'
            sidecar.write_text(json.dumps({'schema_version':1,'status':'failed','reason':reason,
                                         'mesh_file':source.name,'parts':{}},ensure_ascii=False),encoding='utf-8')
            result={'status':'failed','path':str(sidecar),'reason':reason}
            print(f'WARNING: body-part file failed; reconstruction retained: {reason}',flush=True)
        result['elapsed_seconds']=round(time.perf_counter()-started,3)
        report['segmentation']=result
    else:
        report['segmentation']={'status':'disabled','reason':'Explicit --no-segmentation; existing position file, if any, is not deleted.'}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def add_geometry_postprocess(report_path, enabled=True, raw_unit='mm', stage='hands'):
    """Apply once; stage PLY/OBJ/positions together and roll back on export errors."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    field={'surface': 'surface_quality', 'skeleton': 'skeleton_hand_refinement',
           'skeleton_density': 'hand_density_refinement', 'hands_auto': 'hand_processing',
           'palm_volume': 'palm_volume_trial', 'palm_focus': 'palm_focus_trial', 'palm_reference': 'palm_reference_trial', 'palm_whole': 'palm_whole_trial', 'finger_tube': 'finger_tube_trial', 'hands': 'hand_refinement'}[stage]
    if not enabled:
        report[field] = {'status': 'disabled', 'applied': False}
    elif report.get(field, {}).get('status') == 'complete':
        print(f'{stage}: reusing completed result (no repeated smoothing).', flush=True)
        return report
    elif report.get('segmentation', {}).get('status') != 'complete':
        report[field] = {'status': 'skipped', 'applied': False, 'reason': 'Body-part positions unavailable'}
    else:
        started = time.perf_counter()
        mesh_path = report_path.parent/report['artifacts']['ply']['name']
        final_mesh = report_path.parent/(Path(report['source']).stem+'_mesh.ply')
        final_obj = final_mesh.with_suffix('.obj')
        positions = report_path.parent/'body_parts.json'
        try:
            print(f'\n[{stage}: bounded geometry candidate and quality checks]', flush=True)
            original_hash = file_hash(mesh_path)
            centerlines = None
            if stage=='surface':
                mesh=o3d.io.read_triangle_mesh(str(mesh_path))
                raw=np.asarray(o3d.io.read_point_cloud(report['source']).points).copy()
                raw=raw[np.isfinite(raw).all(1)]*(1000 if raw_unit=='m' else 1)
                data=json.loads(positions.read_text(encoding='utf-8'))
                if data['mesh']['sha256']!=original_hash:raise ValueError('Positions do not match the mesh')
                candidate,result=refine_surface(mesh,raw,data['parts'])
            elif stage in ('palm_volume', 'palm_focus', 'palm_reference', 'palm_whole', 'finger_tube'):
                from palm_volume import palm_candidate
                centerlines = json.loads((report_path.parent/'hand_centerlines.json').read_text(encoding='utf-8'))
                if original_hash not in (centerlines.get('source_mesh_sha256'), centerlines.get('output_mesh_sha256')):
                    raise ValueError('Palm trial centerlines do not match input mesh.')
                data = json.loads(positions.read_text(encoding='utf-8'))
                if data['mesh']['sha256'] != original_hash:
                    raise ValueError('Palm trial labels do not match input mesh.')
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                raw = np.asarray(o3d.io.read_point_cloud(report['source']).points).copy()
                raw = raw[np.isfinite(raw).all(1)]*(1000 if raw_unit == 'm' else 1)
                if stage == 'finger_tube':
                    from finger_envelope import constrain_fingers
                    candidate, result = constrain_fingers(mesh, raw, data, centerlines['parts'])
                else:
                    candidate, result = palm_candidate(mesh, raw, data, centerlines['parts'],
                        palm_focus=(stage == 'palm_focus'), reference_first=(stage == 'palm_reference'), whole_palm=(stage == 'palm_whole'))
            elif stage in ('skeleton', 'skeleton_density', 'hands_auto'):
                raw = np.asarray(o3d.io.read_point_cloud(report['source']).points).copy()
                raw = raw[np.isfinite(raw).all(1)]*(1000 if raw_unit == 'm' else 1)
                candidate, result, centerlines = refine_skeleton_hands(mesh_path, positions, raw,
                    density_constraints=(stage != 'skeleton'), conservative=(stage == 'hands_auto'))
            else:candidate, result = refine_hands(mesh_path, positions)
            if result['applied']:
                data = json.loads(positions.read_text(encoding='utf-8'))
                with tempfile.TemporaryDirectory(prefix='.hands_', dir=report_path.parent) as scratch:
                    scratch = Path(scratch)
                    new_mesh = scratch/final_mesh.name; new_obj = scratch/final_obj.name; new_positions = scratch/positions.name
                    if not o3d.io.write_triangle_mesh(str(new_mesh), candidate, write_vertex_normals=True, write_vertex_colors=True):
                        raise OSError('Hand candidate PLY export failed')
                    convert_ply_to_obj(new_mesh, new_obj, vertex_colors=True)
                    vertices = np.asarray(o3d.io.read_triangle_mesh(str(new_mesh)).vertices)
                    # Labels follow unchanged vertex indices; refresh all geometric
                    # fields/hash, explicitly recording that labels were transported.
                    for part in data['parts'].values():
                        ids = part['vertex_indices']
                        if ids:
                            points = vertices[ids]
                            part['center_mm'] = points.mean(0).tolist()
                            part['bbox_mm'] = {'min': points.min(0).tolist(), 'max': points.max(0).tolist()}
                    digest = file_hash(new_mesh)
                    data['mesh']['sha256'] = digest
                    data['mesh']['file'] = final_mesh.name
                    data['fingerprint']['mesh_sha256'] = digest
                    data['geometry_update'] = {'method': result['method'], 'source_mesh_sha256': original_hash,
                                               'labels_transported_by_unchanged_vertex_indices': True}
                    data['notes'].append(f'{stage} geometry updated after inference; labels transported with stable indices, centers/bounds/hash refreshed.')
                    new_positions.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
                    files = [(new_obj, final_obj), (new_mesh, final_mesh), (new_positions, positions)]
                    existing={}
                    for index, (_, destination) in enumerate(files):
                        existing[index]=destination.exists()
                        if existing[index]:shutil.copyfile(destination, scratch/f'backup_{index}')
                    try:
                        for staged, destination in files: publish_with_inherited_permissions(staged,destination)
                    except Exception:
                        for index, (_, destination) in enumerate(files):
                            if existing[index]:publish_with_inherited_permissions(scratch/f'backup_{index}',destination)
                            else:destination.unlink(missing_ok=True)
                        raise
                for key, path in [('ply', final_mesh), ('obj', final_obj)]:
                    report['artifacts'][key]={'name':path.name,'sha256':file_hash(path)}
                report['segmentation']['sha256'] = file_hash(positions)
                report['segmentation']['labels_transported_after_'+field] = True
                xyz = (np.asarray(candidate.vertices)-candidate.get_center())*.001
                faces = np.asarray(candidate.triangles)
                report['appearance']['exported_centered_signed_flux_m3'] = float(np.einsum(
                    'ij,ij->i', xyz[faces[:, 0]], np.cross(xyz[faces[:, 1]], xyz[faces[:, 2]])).sum()/6)
                if result.get('topology_changed'):
                    topology_mesh = copy.deepcopy(candidate)
                    topology_mesh.scale(.001, center=np.zeros(3))
                    report['final_topology'] = geometry_report(topology_mesh, check_intersections=False)
                    report['final_topology']['updated_after'] = field
            if centerlines is not None:
                centerlines['output_mesh_sha256'] = report['artifacts']['ply']['sha256']
                sidecar = report_path.parent/'hand_centerlines.json'
                sidecar.write_text(json.dumps(centerlines, ensure_ascii=False, indent=2), encoding='utf-8')
                result['centerlines'] = {'name': sidecar.name, 'sha256': file_hash(sidecar)}
            result['elapsed_seconds'] = round(time.perf_counter()-started, 3)
            report[field] = result
            print(f"Hands: {'applied' if result['applied'] else 'original retained'}; "
                  f"method={result['method']}; "
                  f"new local intersections={result.get('new_local_intersections', result.get('new_intersections',0))}", flush=True)
        except Exception as error:
            report[field] = {'status': 'failed', 'applied': False, 'reason': str(error)}
            print(f'WARNING: hand refinement skipped: {error}', flush=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def add_hand_refinement(report_path,enabled=True,raw_unit='mm'):
    return add_geometry_postprocess(report_path,enabled,raw_unit,'hands')


def add_surface_quality(report_path,enabled=True,raw_unit='mm'):
    return add_geometry_postprocess(report_path,enabled,raw_unit,'surface')


def add_skeleton_hand_refinement(report_path, enabled=False, raw_unit='mm', *, density_constraints=False):
    return add_geometry_postprocess(report_path, enabled, raw_unit, 'skeleton_density' if density_constraints else 'skeleton')


def add_unified_hand_refinement(report_path, enabled=True, raw_unit='mm'):
    """Default for every new human scan; one evidence/quality policy, no scan IDs."""
    return add_geometry_postprocess(report_path, enabled, raw_unit, 'hands_auto')


def finalize_model_files(report_path):
    """Publish canonical filenames once, including skipped/failed hand stages."""
    report_path=Path(report_path);report=json.loads(report_path.read_text(encoding='utf-8'))
    stem=Path(report['source']).stem
    for key,extension in [('obj','.obj'),('ply','.ply')]:
        source=report_path.parent/report['artifacts'][key]['name']
        target=report_path.parent/(stem+'_mesh'+extension)
        if source.resolve()!=target.resolve():
            shutil.copyfile(source,target)
            report['artifacts'][key]={'name':target.name,'sha256':file_hash(target)}
    positions=report_path.parent/'body_parts.json'
    if positions.exists():
        data=json.loads(positions.read_text(encoding='utf-8'))
        if data.get('status')=='complete':
            data['mesh']['file']=report['artifacts']['ply']['name']
            positions.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
            report['segmentation']['sha256']=file_hash(positions)
    # Pre-hand files are disposable; the base reconstruction remains for review.
    for extension in ('.ply','.obj'):
        temporary=report_path.parent/'intermediate'/(stem+'_pre_hands'+extension)
        try:temporary.unlink(missing_ok=True)
        except OSError:pass
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def distance_check(mesh, sample_m):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distance = scene.compute_distance(o3d.core.Tensor(sample_m.astype(np.float32))).numpy()
    if not np.isfinite(distance).all():
        raise RuntimeError('Nonfinite point-to-mesh distances.')
    return {'median_mm': float(np.median(distance) * 1000),
            'p95_mm': float(np.quantile(distance, .95) * 1000)}


def accept_refinement(before, after):
    # Relative to identical raw queries, not already-denoised observations.
    return (after['p95_mm'] <= before['p95_mm'] + max(.2, .03 * before['p95_mm'])
            and after['median_mm'] <= before['median_mm'] + .1)


def repair_small_topology_defects(mesh, base_report):
    """Try bounded face deletion on a copy; never move surviving vertices.

    Input uses meters. Remove smallest faces at overfull edges, then retain the
    largest surface. Only accept <=0.1% loss of both faces and surface area,
    a single orientable component, and manifold edges/vertices. Small holes
    may remain; this is not hole filling or print certification.
    """
    if base_report.get('edges_with_more_than_two_faces', 0) == 0:
        raise RuntimeError('Orientation failed without overfull edges; automatic cleanup is not applicable.')
    candidate = copy.deepcopy(mesh)
    before_faces = len(mesh.triangles)
    before_area = mesh.get_surface_area()
    candidate.remove_non_manifold_edges()
    if not candidate.has_triangles():
        raise RuntimeError('Topology cleanup would empty the mesh.')
    labels, _, areas = candidate.cluster_connected_triangles()
    candidate.remove_triangles_by_mask(np.asarray(labels) != int(np.argmax(areas)))
    candidate.remove_unreferenced_vertices()
    removed = before_faces - len(candidate.triangles)
    removed_area = max(0., before_area - candidate.get_surface_area())
    if removed > .001 * before_faces or removed_area > .001 * before_area:
        raise RuntimeError('Topology cleanup exceeds the 0.1% face/area budget; intermediate retained.')
    # Keep the dominant existing winding, including open meshes. Geometry QA
    # may normalize a closed mesh; restore the majority direction afterwards.
    reference_faces = np.asarray(candidate.triangles).copy()
    points = np.asarray(candidate.vertices)
    reference_cross = np.cross(points[reference_faces[:, 1]] - points[reference_faces[:, 0]],
                               points[reference_faces[:, 2]] - points[reference_faces[:, 0]])
    checked = geometry_report(candidate, check_intersections=False)
    if (not checked['orientation_succeeded'] or checked['edges_with_more_than_two_faces']
            or not checked['vertex_manifold'] or checked['connected_components'] != 1):
        raise RuntimeError('Limited topology cleanup did not produce an orientable manifold surface.')
    faces = np.asarray(candidate.triangles)
    cross = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                     points[faces[:, 2]] - points[faces[:, 0]])
    agreement = np.sign(np.einsum('ij,ij->i', reference_cross, cross))
    if np.dot(agreement, np.linalg.norm(reference_cross, axis=1)) < 0:
        candidate.triangles = o3d.utility.Vector3iVector(faces[:, [0, 2, 1]].copy())
    candidate.compute_vertex_normals()
    return candidate, {'applied': True, 'method': 'bounded non-manifold edge face cleanup and largest component',
                       'removed_faces': removed, 'removed_face_fraction': removed / before_faces,
                       'removed_area_mm2': removed_area * 1e6,
                       'removed_area_fraction': removed_area / before_area,
                       'vertex_positions_preserved': True,
                       'boundary_edges_before': base_report['boundary_edges'],
                       'boundary_edges_after': checked['boundary_edges'],
                       'checked_topology': checked,
                       'note': 'May leave small openings. Dominant winding retained; no hole filling or self-intersection certification.'}


def reconstruct(source, unit='mm', fast=False, redo=False, sole_axis='off', up_sign=1, refine=False, no_shoe_repair=False, shoe_axis='auto', appearance='full', segmentation=True, seg_python=None, seg_axis='auto', hand_refine=False, kind='auto', surface_refine=True, raw_color=True, face_uv=True, hand_skeleton=False, hand_density=False, hand_auto=True, face_refine=True, hand_model=None, face_model=None, settings_path=None):
    source = Path(source).resolve()
    started = time.perf_counter()
    model_settings = resolve_model_settings(settings_path, hand_model, face_model)
    hand_auto = bool(hand_auto and not any((hand_refine, hand_skeleton, hand_density)))
    mano_enabled = bool(model_settings['requested']['use_mano_hand_model'] and hand_auto and segmentation)
    face_refine = bool(model_settings['requested']['use_flame_face_model'] and face_refine and segmentation and surface_refine and not fast)
    model_settings['effective'] = {'mano': mano_enabled, 'flame': face_refine}
    signature = fingerprint(source, unit, fast)
    signature.update(sole_axis=sole_axis, up_sign=up_sign, refine=refine, no_shoe_repair=no_shoe_repair,
                     shoe_axis=shoe_axis, appearance=appearance, hand_refine=bool(hand_refine and segmentation),kind=kind,surface_refine=bool(surface_refine and segmentation),hand_skeleton=bool(hand_skeleton and segmentation),hand_density=bool(hand_density and segmentation),hand_auto=bool(hand_auto and segmentation))
    signature['face_refine'] = face_refine
    signature['model_application'] = model_settings['effective']
    if sum(map(bool, (hand_skeleton, hand_refine, hand_density))) > 1:
        raise ValueError('Choose one hand processing mode.')
    raw_input = {'status': 'skipped', 'reason': 'RAW color disabled by appearance options.'}
    if appearance == 'full' and raw_color:
        try:
            raw_input = raw_color_input(RAW_DATA, source.stem)
        except (OSError, ValueError) as error:
            raw_input = {'status': 'skipped', 'reason': str(error)}
    signature['raw_color_input'] = raw_input
    if kind=='auto':
        signature['subject_detection']={'enabled':segmentation,'python':str(Path(seg_python or DEFAULT_GPU_PYTHON).resolve()),'axis':seg_axis}
    if hand_refine and segmentation:
        signature['hand_segmentation'] = {'python': str(Path(seg_python or DEFAULT_GPU_PYTHON).resolve()), 'axis': seg_axis}
    if surface_refine and segmentation:
        signature['surface_segmentation']={'python':str(Path(seg_python or DEFAULT_GPU_PYTHON).resolve()),'axis':seg_axis}
    if (hand_auto or hand_skeleton or hand_density) and segmentation:
        signature['skeleton_segmentation']={'python':str(Path(seg_python or DEFAULT_GPU_PYTHON).resolve()),'axis':seg_axis}
    if not no_shoe_repair and sole_axis != 'off':
        raise ValueError('Shoe closure cannot be combined with old sole trimming; use --sole-axis off or --no-shoe-repair.')
    folder = OUTPUTS / source.stem
    cached = None if redo else cached_result(folder, signature)
    if cached:
        path, report = cached
        print(f"Unchanged input/code; reusing: {path.parent / report['artifacts']['ply']['name']}", flush=True)
        add_body_part_positions(path,segmentation,seg_python,seg_axis,up_sign)
        add_hand_refinement(path,hand_refine and segmentation and report.get('subject',{}).get('kind')=='human',unit)
        add_surface_quality(path,surface_refine and segmentation and report.get('subject',{}).get('kind')=='human',unit)
        add_skeleton_hand_refinement(path, hand_skeleton and segmentation and report.get('subject',{}).get('kind')=='human', unit)
        add_skeleton_hand_refinement(path, hand_density and segmentation and report.get('subject',{}).get('kind')=='human', unit, density_constraints=True)
        add_unified_hand_refinement(path, hand_auto and segmentation and report.get('subject',{}).get('kind')=='human', unit)
        report = apply_mano(path, mano_enabled)
        report = apply_scan_polish(path, hand_auto and segmentation, surface_refine and segmentation, unit, seg_axis, up_sign)
        report = apply_face_refinement(path, face_refine, seg_axis, up_sign)
        report = finalize_model_files(path)
        report['model_settings'] = model_settings
        report['face_uv'] = export_face_uv(path.parent, report, face_uv and not fast)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if report['face_uv']['status'] == 'complete':
            print(f"Textured OBJ: {path.parent / report['face_uv']['obj']}", flush=True)
        return path
    cloud = o3d.io.read_point_cloud(str(source))
    raw_color_report = dict(raw_input)
    if raw_input['status'] == 'ready':
        try:
            raw_color_report = apply_raw_colors(cloud, unit, raw_input)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as error:
            raw_color_report = {'status': 'skipped', 'source': raw_input['folder'],
                                'frame': raw_input['frame'], 'reason': str(error)}
    if raw_color_report['status'] == 'used':
        print(f"RAW color: repaired {raw_color_report['repaired_points']:,} source points "
              f"from frame {raw_color_report['frame']}.", flush=True)
    elif appearance == 'full' and raw_color:
        print(f"RAW color skipped: {raw_color_report['reason']}", flush=True)
    cloud.remove_non_finite_points()
    cloud.remove_duplicated_points()
    if unit == 'm':
        cloud.scale(1000, center=np.zeros(3))
    points_mm = np.asarray(cloud.points)
    settings = choose_parameters(points_mm, fast)
    print(f"\nInput: {source.name} | spacing={settings['sampled_spacing_mm']:.2f} mm | "
          f"spread proxy={settings['local_spread_proxy_mm']:.2f} mm\n"
          f"Auto: voxel={settings['voxel_mm']} mm, depth={settings['depth']}, "
          f"normal radius={settings['normal_radius_mm']} mm, point passes={settings['point_iterations']}", flush=True)
    run_dir = folder / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', str(ROOT / 'reconstruct_human.py'), str(source),
               '--out', str(run_dir / 'intermediate'), '--input-unit', unit, '--output-unit', 'mm',
               '--preset', 'scan', '--no-vis', '--skip-self-intersections',
               '--voxel', str(settings['voxel_mm'] / 1000), '--depth', str(settings['depth']),
               '--normal-radius', str(settings['normal_radius_mm'] / 1000),
               '--normal-max-nn', str(settings['normal_neighbors']),
               '--point-denoise-iterations', str(settings['point_iterations']),
               '--point-denoise-max-move-mm', str(settings['point_move_cap_mm']),
               '--point-cluster-radius-mm', str(settings['cluster_radius_mm']),
               '--smooth-iterations', str(settings['smooth_iterations']), '--qa-samples', '5000', '--threads', str(WORKERS),
               '--sole-guard', sole_axis if kind=='human' else 'off', '--up-sign', str(up_sign),
               '--sole-margin-mm', str(max(6., settings['voxel_mm'] * 3))]
    subprocess.run(command, check=True)
    base_path = run_dir / 'intermediate' / f'{source.stem}_mesh.ply'
    base_report = json.loads((run_dir / 'intermediate' / f'{source.stem}_report.json').read_text(encoding='utf-8'))
    mesh = o3d.io.read_triangle_mesh(str(base_path))
    if kind=='auto' and segmentation:
        probe=run_dir/'intermediate'/'subject_probe.json'
        ensure_body_parts(base_path,gpu_python=seg_python,axis=seg_axis,up_sign=up_sign,output=probe)
        subject=classify_parts(json.loads(probe.read_text(encoding='utf-8')))
    else:
        subject={'kind':kind if kind!='auto' else 'unknown','reason':'Explicit type override.' if kind!='auto' else 'Recognition disabled; generic fallback.'}
    human=subject['kind']=='human'
    print(f"Subject: {subject['kind']}; {subject['reason']}",flush=True)
    object_trim={'applied':False}
    if not human:mesh,object_trim=generic_observation_trim(mesh,points_mm,settings['voxel_mm'])
    mesh.scale(.001, center=np.zeros(3))
    topology_repair = {'applied': False, 'reason': 'Base triangle orientation succeeded.'}
    if not human:
        mesh,topology_repair=repair_generic_topology(mesh)
    elif not base_report['orientation_succeeded']:
        print('WARNING: base orientation failed; attempting bounded topology cleanup.', flush=True)
        try:
            mesh, topology_repair = repair_small_topology_defects(mesh, base_report)
        except RuntimeError as error:
            if not human:
                topology_repair={'applied':False,'reason':str(error),'status':'generic_preview_with_topology_defects',
                                 'note':'Export for viewing only; not certified manifold or watertight.'}
                print(f'WARNING: generic mesh exported with recorded topology defects: {error}',flush=True)
            else:
                failure_path = run_dir / 'failure.json'
                failure_path.write_text(json.dumps({'status': 'failed', 'source': str(source),
                                                   'error': str(error), 'intermediate': str(base_path)},
                                                  ensure_ascii=False, indent=2), encoding='utf-8')
                raise RuntimeError(f'{error} Details: {failure_path}') from error
        if topology_repair['applied']:
            print(f"Topology cleanup: removed {topology_repair['removed_faces']} faces; vertex positions preserved.",flush=True)
    mesh.compute_vertex_normals()
    # Work near the origin for float32 distance checks; restore it on export.
    origin = np.asarray(mesh.get_center()).copy()
    mesh.translate(-origin)
    cloud.scale(.001, center=np.zeros(3))
    cloud.translate(-origin)
    raw = np.asarray(cloud.points)
    sample = raw[np.random.default_rng(7).choice(len(raw), min(6000, len(raw)), replace=False)]
    print('\n[observation diagnostics; extra denoise is opt-in]', flush=True)
    _, labels, _, _, support = observation_support(mesh, raw, workers=WORKERS)
    active = np.where(labels == 2, 0., np.where(labels == 1, .5, 1.))
    before = distance_check(mesh, sample)
    candidate, movement = (bilateral_mesh_filter(mesh, normal_iterations=4, vertex_iterations=8,
                                                max_move=settings['mesh_move_cap_mm'] / 1000,
                                                active_weights=active) if refine else
                           (mesh, {'method': 'skipped; validated baseline retained', 'movement_max_mm': 0.}))
    after = distance_check(candidate, sample) if refine else before.copy()
    accepted = refine and accept_refinement(before, after)
    final = candidate if accepted else copy.deepcopy(mesh)
    print(f"Raw-observation P95: {before['p95_mm']:.3f} -> {after['p95_mm']:.3f} mm; "
          f"refinement {'accepted' if accepted else 'not applied, using base mesh'}", flush=True)
    topology = geometry_report(final, check_intersections=False)
    transfer_colors(final, cloud, np.zeros(3), 1, WORKERS)
    final.translate(origin)
    final.scale(1000, center=np.zeros(3))
    plan = shoe_plan(base_report, signature['source_sha256'] if unit == 'mm' else '', human and not no_shoe_repair, shoe_axis, up_sign)
    if not human:plan={'enabled':False,'reason':'Generic object branch; shoe cutting disabled.'}
    shoe_report = dict(plan)
    if plan['enabled']:
        print(f"Shoe repair: {plan['profile']}, axis={plan['axis']}, up={up_sign}, floor={plan['floor_up_mm']:.3f} mm", flush=True)
        try:
            repaired, shoe_report = apply_shoe_plan(final, plan)
            if shoe_report['enabled']:
                # Check a copy: don't change the export's history-compatible winding.
                checked = geometry_report(copy.deepcopy(repaired).scale(.001, center=np.zeros(3)), check_intersections=False)
                if (not checked['orientation_succeeded'] or checked['boundary_edges'] > topology['boundary_edges']
                        or checked['edges_with_more_than_two_faces'] > topology['edges_with_more_than_two_faces']):
                    raise RuntimeError('Shoe closure worsened orientation or edge topology')
                final, topology = repaired, checked
                shoe_report['topology_checked_on_copy_export_winding_preserved'] = True
                print(f"Shoe repair applied, cap faces={shoe_report['cap_triangles']}", flush=True)
        except (ValueError, RuntimeError, IndexError) as error:
            shoe_report = {**plan, 'enabled': False, 'reason': str(error), 'fallback_to_base': True}
        if not shoe_report['enabled']:
            print(f"WARNING: shoe repair skipped; original mesh retained: {shoe_report['reason']}", flush=True)
    elif not no_shoe_repair and human:
        print(f"WARNING: shoe repair skipped: {shoe_report['reason']}", flush=True)
    # Calibrated RAW can distinguish missing colors from observed dark material.
    # With RAW colors, keep those observations instead of guessing from brightness.
    appearance_mode = 'normals' if raw_color_report['status'] == 'used' else appearance
    print(f'\n[appearance: {appearance_mode}]', flush=True)
    appearance_axis = shoe_axis if shoe_axis != 'auto' else sole_axis if sole_axis != 'off' else 'auto'
    try:
        # Final coordinates are always mm, even for --unit m input. Run after
        # color transfer and shoe closure; later stages must not reset RGB or winding.
        final, appearance_report = apply_appearance(final, mode=appearance_mode, up_axis=appearance_axis,
                                                    up_sign=up_sign, unit='mm',color_region='body' if human else 'all')
    except (ValueError, RuntimeError) as error:
        appearance_report = {'enabled': False, 'mode': appearance_mode, 'reason': str(error),
                             'fallback_to_previous_mesh': True}
        print(f'WARNING: appearance processing skipped; previous mesh retained: {error}', flush=True)
    if appearance_report['enabled']:
        normal_result = appearance_report['normals']
        color_result = appearance_report['color_repair']
        print(f"Appearance: flipped {normal_result.get('flipped_faces', 0)} faces; "
              f"repaired RGB on {color_result.get('repaired_vertices', 0)} vertices.", flush=True)
        for name, result in [('normals', normal_result), ('color', color_result)]:
            if not result.get('applied') and result.get('reason') != 'Explicitly disabled.':
                print(f"Appearance {name} skipped: {result.get('reason')}", flush=True)
    # Edge counts remain valid because appearance preserves connectivity.
    # Distinguish the earlier QA winding from the actual exported orientation.
    topology['global_winding']['stage'] = 'topology_qa_before_appearance'
    centered = (np.asarray(final.vertices) - final.get_center()) * .001
    triangles = np.asarray(final.triangles)
    appearance_report['exported_centered_signed_flux_m3'] = float(np.einsum(
        'ij,ij->i', centered[triangles[:, 0]],
        np.cross(centered[triangles[:, 1]], centered[triangles[:, 2]])).sum() / 6)
    # Build privately before publishing final files: no rewrite of a displayed OBJ.
    output = run_dir / 'intermediate' / f'{source.stem}_pre_hands.ply'
    if not o3d.io.write_triangle_mesh(str(output), final, write_vertex_normals=True, write_vertex_colors=True):
        raise OSError(f'Failed to write {output}')
    obj = output.with_suffix('.obj')
    convert_ply_to_obj(output, obj, vertex_colors=True)
    if file_hash(source) != signature['source_sha256']:
        raise RuntimeError('Source changed during reconstruction; result is not cached. Rerun when copying has finished.')
    report = {'fingerprint': signature, 'source': str(source), 'auto_settings': settings,'subject':subject,'object_processing':object_trim,
              'model_settings': model_settings,
              'unit': 'mm', 'raw_queries': len(sample), 'before_raw_distance': before,
              'candidate_raw_distance': after, 'refinement_accepted': accepted,
              'candidate_movement': movement, 'base_observation_support': support,
              'final_topology': topology, 'base_report': base_report, 'shoe_repair': shoe_report,
              'base_topology_repair': topology_repair,
              'appearance': appearance_report, 'raw_color': raw_color_report,
              'notes': ['Validated scan baseline; no guarantee of best/true face reconstruction.',
                        'Extra support-weighted refinement is opt-in, not enabled by default.',
                        'Raw distances are observation fit, not ground-truth accuracy.',
                        'Self-intersection check skipped; watertightness/print readiness unknown.',
                        'OBJ RGB is a viewer-dependent vertex-color extension, not a texture.'],
              'total_seconds': round(time.perf_counter() - started, 3),
              'artifacts': {key: {'name': str(path.relative_to(run_dir)), 'sha256': file_hash(path)}
                            for key, path in [('ply', output), ('obj', obj), ('base', base_path)]}}
    report_path = run_dir / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # Store positions from the exact final exported PLY, after shoe/appearance
    # processing. Segmentation options deliberately do not invalidate mesh cache.
    report=add_body_part_positions(report_path,segmentation,seg_python,seg_axis,up_sign)
    report=add_hand_refinement(report_path,hand_refine and segmentation and human,unit)
    report=add_surface_quality(report_path,surface_refine and segmentation and human,unit)
    report=add_skeleton_hand_refinement(report_path, hand_skeleton and segmentation and human, unit)
    report=add_skeleton_hand_refinement(report_path, hand_density and segmentation and human, unit, density_constraints=True)
    report=add_unified_hand_refinement(report_path, hand_auto and segmentation and human, unit)
    report=apply_mano(report_path, mano_enabled)
    report=apply_scan_polish(report_path, hand_auto and segmentation, surface_refine and segmentation, unit, seg_axis, up_sign)
    report=apply_face_refinement(report_path, face_refine, seg_axis, up_sign)
    report=finalize_model_files(report_path)
    report['face_uv'] = export_face_uv(run_dir, report, face_uv and not fast)
    output=run_dir/report['artifacts']['ply']['name'];obj=run_dir/report['artifacts']['obj']['name']
    report['total_seconds']=round(time.perf_counter()-started,3)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"\nDONE in {report['total_seconds']:.1f} s\nPLY: {output}\nOBJ: {obj}\nReport: {report_path}", flush=True)
    if report['face_uv']['status'] == 'complete':
        print(f"Textured OBJ: {run_dir / report['face_uv']['obj']}", flush=True)
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', dest='settings_path', help='Model settings JSON; default: project settings.json.')
    hand_model_options = parser.add_mutually_exclusive_group()
    hand_model_options.add_argument('--hand-model', dest='hand_model', action='store_true', help='Enable MANO application, overriding settings.json.')
    hand_model_options.add_argument('--no-hand-model', dest='hand_model', action='store_false', help='Disable MANO application; retain ordinary hand processing.')
    face_model_options = parser.add_mutually_exclusive_group()
    face_model_options.add_argument('--face-model', dest='face_model', action='store_true', help='Enable FLAME application, overriding settings.json.')
    face_model_options.add_argument('--no-face-model', dest='face_model', action='store_false', help='Disable FLAME application; retain ordinary surface processing.')
    parser.set_defaults(hand_model=None, face_model=None)
    parser.add_argument('input', nargs='?', help='Number (e.g. 3), filename, or PLY path; default: latest modified PLY in data.')
    parser.add_argument('--all', action='store_true', help='Process all PLY inputs, reusing unchanged completed results.')
    parser.add_argument('--redo', action='store_true', help='Rebuild into a new directory even if a cached result exists.')
    parser.add_argument('--fast', action='store_true', help='Coarser quick preview; lower fine-detail resolution.')
    parser.add_argument('--kind',choices=['auto','human','object'],default='auto',help='Auto: require coherent human parts, otherwise generic object processing.')
    parser.add_argument('--no-surface-refine',action='store_true',help='Disable quality-gated face/body polishing and torso quadratic denoising; hand processing has its own switch.')
    parser.add_argument('--no-face-refine', action='store_true', help='Disable the FLAME face patch step; retain other surface processing.')
    parser.add_argument('--unit', choices=['mm', 'm'], default='mm', help='Input coordinate unit; default mm. Outputs always mm.')
    parser.add_argument('--sole-axis', choices=['auto', 'x', 'y', 'z', 'off'], default='off')
    parser.add_argument('--refine', action='store_true', help='Opt in to additional bounded bilateral filtering; default off.')
    parser.add_argument('--no-shoe-repair', action='store_true', help='Disable shoe closure; enabled by default for new-data testing.')
    parser.add_argument('--shoe-axis', choices=['auto', 'x', 'y', 'z'], default='auto', help='Shoe vertical axis; assumes positive up unless --up-sign -1.')
    parser.add_argument('--up-sign', type=int, choices=[-1, 1], default=1)
    appearance_modes = parser.add_mutually_exclusive_group()
    appearance_modes.add_argument('--no-appearance', dest='appearance', action='store_const', const='off',
                                  help='Disable final normal-direction and dark-patch processing.')
    appearance_modes.add_argument('--normals-only', dest='appearance', action='store_const', const='normals',
                                  help='Correct final normal direction, retaining original vertex RGB.')
    parser.set_defaults(appearance='full')
    parser.add_argument('--no-raw-color', action='store_true', help='Use only the merged PLY for color processing.')
    parser.add_argument('--no-face-uv', action='store_true', help='Skip the optional textured OBJ export while retaining RAW vertex color repair.')
    parser.add_argument('--no-segmentation',action='store_true',help='Skip body_parts.json and hand refinement.')
    hand_modes=parser.add_mutually_exclusive_group()
    hand_modes.add_argument('--hand-refine',dest='hand_refine',action='store_true',help='Opt in to the withdrawn experimental palm geometry processing.')
    hand_modes.add_argument('--no-hand-refine',dest='hand_auto',action='store_false',help='Disable the default automatic hand geometry checks and processing.')
    hand_modes.add_argument('--hand-skeleton', action='store_true', help='Try centerline-guided finger surface fitting without changing face connections.')
    hand_modes.add_argument('--hand-density', action='store_true', help='Try centerlines plus adaptive bridge rejection and observed thickness; may leave openings.')
    parser.set_defaults(hand_refine=False, hand_auto=True)
    parser.add_argument('--seg-python',help='Python executable with GPU PyTorch for Sapiens; defaults to HUMAN_SEG_PYTHON or local test0.')
    parser.add_argument('--seg-axis',choices=['auto','x','y','z'],default='auto',help='Segmentation vertical axis; uses --up-sign.')
    args = parser.parse_args(argv)
    if args.input and args.all:
        parser.error('Use an input OR --all, not both.')
    try:
        inputs = select_inputs(args.input, args.all)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')
    failures = []
    for source in inputs:
        try:
            reconstruct(source, args.unit, args.fast, args.redo, args.sole_axis, args.up_sign,
                        args.refine, args.no_shoe_repair, args.shoe_axis, args.appearance,
                        not args.no_segmentation,args.seg_python,args.seg_axis,args.hand_refine,args.kind,
                        not args.no_surface_refine,raw_color=not args.no_raw_color,face_uv=not args.no_face_uv,hand_skeleton=args.hand_skeleton,hand_density=args.hand_density,hand_auto=args.hand_auto,face_refine=not args.no_face_refine,hand_model=args.hand_model,face_model=args.face_model,settings_path=args.settings_path)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f'FAILED {source.name}: {error}', file=sys.stderr, flush=True)
            failures.append(source.name)
    if failures:
        parser.exit(1, f'Failed inputs: {", ".join(failures)}\n')


if __name__ == '__main__':
    main()
