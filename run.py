"""日常入口：自动管理输入输出，以已验证的 scan 参数重建并导出 PLY/OBJ。
  python run.py              处理 data 中最新修改的 PLY
  python run.py 3            处理 export_00003.ply，也可直接写 PLY 路径
  --all：处理全部数据；--redo：忽略缓存重建；--fast：粗略快速预览
  --unit m：输入为米（默认 mm）；--sole-axis x：显式开启鞋底约束并指定竖直轴
  --up-sign -1：该轴负向朝上；--sole-axis off：关闭鞋底外扩约束
  默认关闭鞋底删面及额外双边过滤；--refine 可显式开启额外过滤。
  默认尝试鞋底截断补面（新数据为测试功能）；--no-shoe-repair 可关闭。
  --shoe-axis x/y/z：手动指定鞋底竖直轴，默认 auto 最长轴，--up-sign -1 为负向朝上。
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
import time
from datetime import datetime

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from mesh_refine import bilateral_mesh_filter, observation_support
from ply_to_obj import convert_ply_to_obj
from reconstruct_human import geometry_report, transfer_colors
from sole_repair import shoe_plan, apply_shoe_plan

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs' / 'models'
WORKERS = min(8, os.cpu_count() or 1)
VERSION = 5


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


def fingerprint(source, unit, fast):
    code = {name: file_hash(ROOT / name) for name in
            ('run.py', 'reconstruct_human.py', 'surface_detail.py', 'mesh_refine.py', 'ply_to_obj.py', 'sole_repair.py')}
    return {'version': VERSION, 'source_sha256': file_hash(source), 'unit': unit,
            'fast': fast, 'code_sha256': code, 'open3d_version': o3d.__version__,
            'numpy_version': np.__version__}


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
            if valid:
                return report_path, report
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


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


def reconstruct(source, unit='mm', fast=False, redo=False, sole_axis='off', up_sign=1, refine=False, no_shoe_repair=False, shoe_axis='auto'):
    source = Path(source).resolve()
    started = time.perf_counter()
    signature = fingerprint(source, unit, fast)
    signature.update(sole_axis=sole_axis, up_sign=up_sign, refine=refine, no_shoe_repair=no_shoe_repair, shoe_axis=shoe_axis)
    if not no_shoe_repair and sole_axis != 'off':
        raise ValueError('Shoe closure cannot be combined with old sole trimming; use --sole-axis off or --no-shoe-repair.')
    folder = OUTPUTS / source.stem
    cached = None if redo else cached_result(folder, signature)
    if cached:
        path, report = cached
        print(f"Unchanged input/code; reusing: {path.parent / report['artifacts']['ply']['name']}", flush=True)
        return path
    cloud = o3d.io.read_point_cloud(str(source))
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
               '--sole-guard', sole_axis, '--up-sign', str(up_sign),
               '--sole-margin-mm', str(max(6., settings['voxel_mm'] * 3))]
    subprocess.run(command, check=True)
    base_path = run_dir / 'intermediate' / f'{source.stem}_mesh.ply'
    base_report = json.loads((run_dir / 'intermediate' / f'{source.stem}_report.json').read_text(encoding='utf-8'))
    if not base_report['orientation_succeeded']:
        raise RuntimeError('Triangle orientation failed. Intermediate saved for inspection; not exporting it as a successful final model.')
    mesh = o3d.io.read_triangle_mesh(str(base_path))
    mesh.scale(.001, center=np.zeros(3))
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
    plan = shoe_plan(base_report, signature['source_sha256'] if unit == 'mm' else '', not no_shoe_repair, shoe_axis, up_sign)
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
    elif not no_shoe_repair:
        print(f"WARNING: shoe repair skipped: {shoe_report['reason']}", flush=True)
    output = run_dir / f'{source.stem}_mesh.ply'
    if not o3d.io.write_triangle_mesh(str(output), final, write_vertex_normals=True, write_vertex_colors=True):
        raise OSError(f'Failed to write {output}')
    obj = output.with_suffix('.obj')
    convert_ply_to_obj(output, obj, vertex_colors=True)
    if file_hash(source) != signature['source_sha256']:
        raise RuntimeError('Source changed during reconstruction; result is not cached. Rerun when copying has finished.')
    report = {'fingerprint': signature, 'source': str(source), 'auto_settings': settings,
              'unit': 'mm', 'raw_queries': len(sample), 'before_raw_distance': before,
              'candidate_raw_distance': after, 'refinement_accepted': accepted,
              'candidate_movement': movement, 'base_observation_support': support,
              'final_topology': topology, 'base_report': base_report, 'shoe_repair': shoe_report,
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
    print(f"\nDONE in {report['total_seconds']:.1f} s\nPLY: {output}\nOBJ: {obj}\nReport: {report_path}", flush=True)
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', help='Number (e.g. 3), filename, or PLY path; default: latest modified PLY in data.')
    parser.add_argument('--all', action='store_true', help='Process all PLY inputs, reusing unchanged completed results.')
    parser.add_argument('--redo', action='store_true', help='Rebuild into a new directory even if a cached result exists.')
    parser.add_argument('--fast', action='store_true', help='Coarser quick preview; lower fine-detail resolution.')
    parser.add_argument('--unit', choices=['mm', 'm'], default='mm', help='Input coordinate unit; default mm. Outputs always mm.')
    parser.add_argument('--sole-axis', choices=['auto', 'x', 'y', 'z', 'off'], default='off')
    parser.add_argument('--refine', action='store_true', help='Opt in to additional bounded bilateral filtering; default off.')
    parser.add_argument('--no-shoe-repair', action='store_true', help='Disable shoe closure; enabled by default for new-data testing.')
    parser.add_argument('--shoe-axis', choices=['auto', 'x', 'y', 'z'], default='auto', help='Shoe vertical axis; assumes positive up unless --up-sign -1.')
    parser.add_argument('--up-sign', type=int, choices=[-1, 1], default=1)
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
            reconstruct(source, args.unit, args.fast, args.redo, args.sole_axis, args.up_sign, args.refine, args.no_shoe_repair, args.shoe_axis)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f'FAILED {source.name}: {error}', file=sys.stderr, flush=True)
            failures.append(source.name)
    if failures:
        parser.exit(1, f'Failed inputs: {", ".join(failures)}\n')


if __name__ == '__main__':
    main()
