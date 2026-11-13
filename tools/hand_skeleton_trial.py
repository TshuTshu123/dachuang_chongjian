"""复用既有完整模型，调用 run.py 同一个骨架手部步骤，隔离比较实际几何。

项目根目录：python tools/hand_skeleton_trial.py --scan 0 1 2 3
默认验证统一自动处理，输出 hand_auto；--mode density 复现允许开口的密度实验。
--mode palm 单独对比掌部整体限厚，日常不需要运行此试验入口。
--mode tube 试验 12 mm 手指包络，几何检查失败时保留基准，不默认启用。
--mode focus 单独试验曲面锚定的掌心收紧，保护远端手指，不默认启用。
--mode whole 将参考面扩到整掌并连接腕口，左右采用相同规则。
--mode reference 先限定掌部参考包络，排除外部点，再作最多 1 mm 拟合。
仅保留局部手部预览、对比和报告。
--keep-full 3 保留 00003 的完整 PLY/OBJ。
基准来自上一轮 hand_branches/report.json，并核对原始点云、模型和标签哈希。
不重复运行未变化的 Poisson/识别步骤，不覆盖 outputs/models 中的历史结果。
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from body_parts import file_hash
from hand_centerlines import hand_frame
from run import add_skeleton_hand_refinement, add_unified_hand_refinement, add_geometry_postprocess, publish_with_inherited_permissions
from tools.surface_trials import panel


def trial(scan, keep_full=False, mode='auto'):
    started = time.perf_counter()
    baseline = json.loads((ROOT/'outputs/experiments/hand_branches'/f'export_{scan:05d}'/'report.json').read_text())
    source = Path(baseline['source_mesh'])
    raw = Path(baseline['source_cloud'])
    if file_hash(source) != baseline['source_sha256'] or file_hash(raw) != baseline['source_cloud_sha256']:
        raise ValueError('Trial baseline source changed.')
    original_report = json.loads((source.parent/'report.json').read_text(encoding='utf-8'))
    parts_file = source.parent/'body_parts.json'
    original_parts_hash = file_hash(parts_file)
    info = json.loads(parts_file.read_text(encoding='utf-8'))
    out = ROOT/'outputs/experiments'/({'auto': 'hand_auto', 'density': 'hand_density_refine', 'palm': 'palm_volume', 'focus': 'palm_focus', 'reference': 'palm_reference', 'whole': 'palm_whole', 'tube': 'finger_tube'}[mode])/f'export_{scan:05d}'
    out.mkdir(parents=True, exist_ok=True)
    work = out/f'.stage_run_{uuid4().hex}'
    work.mkdir()
    try:
        shutil.copyfile(source, work/'input.ply')
        shutil.copyfile(parts_file, work/'body_parts.json')
        stage = copy.deepcopy(original_report)
        stage.pop('skeleton_hand_refinement', None)
        stage.pop('hand_density_refinement', None)
        stage.pop('hand_processing', None)
        stage.pop('palm_volume_trial', None)
        stage.pop('finger_tube_trial', None)
        stage.pop('palm_focus_trial', None)
        stage.pop('palm_reference_trial', None)
        stage.pop('palm_whole_trial', None)
        stage['artifacts'] = {'ply': {'name': 'input.ply', 'sha256': baseline['source_sha256']},
                              'obj': {'name': 'input.obj', 'sha256': None}}
        report_path = work/'report.json'
        report_path.write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding='utf-8')
        unit = original_report.get('fingerprint', {}).get('unit', 'mm')
        if mode in ('palm', 'tube', 'focus', 'reference', 'whole'):
            shutil.copyfile(ROOT/'outputs/experiments/hand_auto'/f'export_{scan:05d}'/'hand_centerlines.json', work/'hand_centerlines.json')
            field = {'palm': 'palm_volume', 'tube': 'finger_tube', 'focus': 'palm_focus', 'reference': 'palm_reference', 'whole': 'palm_whole'}[mode]
            stage = add_geometry_postprocess(report_path, True, unit, field)
            result = stage[field+'_trial']
        elif mode == 'auto':
            stage = add_unified_hand_refinement(report_path, True, unit)
            result = stage['hand_processing']
        else:
            stage = add_skeleton_hand_refinement(report_path, True, unit, density_constraints=True)
            result = stage['hand_density_refinement']
        if result['status'] != 'complete':
            raise RuntimeError(result)
        candidate_path = work/stage['artifacts']['ply']['name']
        before = o3d.io.read_triangle_mesh(str(source))
        candidate = o3d.io.read_triangle_mesh(str(candidate_path))
        before.compute_vertex_normals(); candidate.compute_vertex_normals()
        v, cv = np.asarray(before.vertices), np.asarray(candidate.vertices)
        assert len(before.vertices) == len(candidate.vertices)
        assert len(candidate.triangles) <= len(before.triangles)
        assert np.array_equal(before.vertex_colors, candidate.vertex_colors)
        parts = info['parts']
        canvas = Image.new('RGB', (1400, 1520), 'white')
        draw = ImageDraw.Draw(canvas)
        previews = []
        for side, name in enumerate(('left_hand', 'right_hand')):
            ids, arm_class, wrist, axes = hand_frame(v, parts, name)
            local = (v-wrist) @ axes
            low = local[ids].min(0)-[12, 25, 15]
            high = local[ids].max(0)+[12, 12, 15]
            # Crop spatial context, including unlabeled surface. Semantic-only
            # extraction creates artificial holes unrelated to the actual mesh.
            chosen = np.flatnonzero(np.all((local >= low) & (local <= high), axis=1)
                                    & (cKDTree(v[ids]).query(v, workers=4)[0] <= 28))
            first, after = before.select_by_index(chosen), candidate.select_by_index(chosen)
            preview = out/f'{name}_preview.ply'
            cut = any(p['part'] == name and p['density_cut_applied']
                      for p in result.get('density_constraints', {}).get('parts', []))
            if result['applied'] and (cut or np.any(np.linalg.norm(cv[chosen]-v[chosen], axis=1) > 1.e-6)):
                if not o3d.io.write_triangle_mesh(str(preview), after):
                    raise OSError('Local preview export failed.')
                previews.append(preview.name)
            else:
                preview.unlink(missing_ok=True)
            gray_before, gray_after = copy.deepcopy(first), copy.deepcopy(after)
            gray_before.paint_uniform_color([.72, .72, .72]); gray_after.paint_uniform_color([.72, .72, .72])
            center = wrist+axes @ ((low+high)/2)
            span = max(high-low)*1.12
            for angle in (0, 1):
                row = side*2+angle
                right, up = (axes[:, 0], axes[:, 1]) if angle == 0 else (axes[:, 2], axes[:, 1])
                for col, (model, title) in enumerate(zip((first, after, gray_before, gray_after),
                        ('Before RGB', 'After RGB', 'Before geometry', 'After geometry'))):
                    canvas.paste(panel(model, center, right, up, span, 350, colored=(col < 2)), (col*350, row*380+25))
                    draw.text((col*350+4, row*380+5), f'{name} | {title}', fill='black')
        canvas.save(out/'comparison.jpg', quality=95)
        centers = work/'hand_centerlines.json'
        if centers.exists(): publish_with_inherited_permissions(centers, out/centers.name)
        kept = []
        for extension in ('.ply', '.obj'):
            final = out/(raw.stem+'_mesh'+extension)
            if keep_full and result['applied']:
                key = 'ply' if extension == '.ply' else 'obj'
                publish_with_inherited_permissions(work/stage['artifacts'][key]['name'], final)
                kept.append({'name': final.name, 'sha256': file_hash(final)})
            else:
                final.unlink(missing_ok=True)
        summary = {'source_mesh': str(source), 'source_sha256': file_hash(source),
                   'source_cloud': str(raw), 'source_cloud_sha256': file_hash(raw),
                   'pipeline_stage': {'auto': 'run.add_unified_hand_refinement',
                       'density': 'run.add_skeleton_hand_refinement(density_constraints=True)',
                       'palm': 'run.add_geometry_postprocess(stage=palm_volume)',
                       'focus': 'run.add_geometry_postprocess(stage=palm_focus)',
                       'reference': 'run.add_geometry_postprocess(stage=palm_reference)',
                       'whole': 'run.add_geometry_postprocess(stage=palm_whole)',
                       'tube': 'run.add_geometry_postprocess(stage=finger_tube)'}[mode],
                   'mode': mode,
                   'geometry_result': result, 'local_previews': previews, 'full_artifacts': kept,
                   'scope': 'Same upstream model; isolated downstream skeleton stage, not a full Poisson rerun.',
                   'elapsed_seconds': round(time.perf_counter()-started, 2)}
        (out/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        assert file_hash(source) == baseline['source_sha256']
        assert file_hash(raw) == baseline['source_cloud_sha256']
        assert file_hash(parts_file) == original_parts_hash
        print(f'export_{scan:05d}: applied={result["applied"]}; seconds={summary["elapsed_seconds"]}', flush=True)
    finally:
        # Delete only known files under our verified, dedicated scratch directory.
        resolved = work.resolve()
        if resolved.parent != out.resolve() or not resolved.name.startswith('.stage_run_'):
            raise RuntimeError('Unexpected scratch directory.')
        for path in work.iterdir():
            if path.is_file(): path.unlink()
            else: raise RuntimeError(f'Unexpected scratch child: {path.name}')
        work.rmdir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--keep-full', type=int, nargs='*', default=[])
    parser.add_argument('--mode', choices=['auto', 'density', 'palm', 'tube', 'focus', 'reference', 'whole'], default='auto')
    args = parser.parse_args()
    for scan in args.scan:
        trial(scan, scan in args.keep_full, args.mode)
