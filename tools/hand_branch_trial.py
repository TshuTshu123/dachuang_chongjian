"""手掌区域约束、多尺度路径追踪的手指中心线试验，不修改正式网格。

项目根目录：python tools/hand_branch_trial.py --scan 0 1 2 3
--self-check 用已知分支的合成形状检查路径、尺度匹配及拒识逻辑。
输出 outputs/experiments/hand_branches/：对比图、中心线 PLY、坐标及评分 JSON。
评分只表示参数稳定性和观测支撑，不是关节识别准确率或统计概率。
仅依赖正式环境已有的 NumPy、SciPy、Open3D、Pillow，不需要额外权重。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_centerlines import assess, trace_mask, match_cost


def self_check():
    # Known 2D topology: broad palm, separated tubes, and a compact shape with no fingers.
    x, y = np.mgrid[-65:66:1.5, -10:181:1.5]
    origin = np.array([-65., -10.])
    palm = (np.abs(x) <= 44) & (y >= 0) & (y <= 75)
    tubes = []
    for cx, end in zip((-36, -18, 0, 18, 36), (125, 155, 165, 153, 130)):
        tubes.append(((x-cx)**2 <= 6**2) & (y >= 65) & (y <= end))
    for count in (5, 4):
        mask = palm | np.logical_or.reduce(tubes[:count])
        branches, _ = trace_mask(mask, origin, 1.5)
        assert len(branches) == count, (count, len(branches))
        flat = np.c_[x[mask], y[mask]]
        cloud = np.vstack([np.c_[flat, np.full(len(flat), z)] for z in (-5., 5.)])
        candidates, _, _ = assess(cloud)
        supported = sum(b['status'] == 'supported_candidate' for b in candidates)
        assert supported == count, ('density_and_consensus', count, supported)
    compact = ((x/42)**2+((y-45)/45)**2) <= 1
    branches, _ = trace_mask(compact, origin, 1.5)
    assert len(branches) == 0, len(branches)
    a = {'xy': np.array([[0., 0], [0, 30]]), 'tip_xy': np.array([0., 30])}
    assert np.isfinite(match_cost(a, a))
    b = {'xy': np.array([[0., 0], [30, 0]]), 'tip_xy': np.array([30., 0])}
    assert not np.isfinite(match_cost(a, b))
    print('Synthetic checks passed: five tubes, four tubes, compact palm, branch matching.', flush=True)


def trial(scan):
    import open3d as o3d
    from PIL import Image, ImageDraw
    from body_parts import file_hash
    from tools.hand_pose_trial import hand_frame, point_panel
    started = time.perf_counter()
    parent = ROOT/'outputs/models'/f'export_{scan:05d}'
    for report_path in sorted(parent.glob('*/report.json'), reverse=True):
        report = json.loads(report_path.read_text(encoding='utf-8'))
        ply = report_path.parent/report['artifacts']['ply']['name']
        parts_path = report_path.parent/'body_parts.json'
        if ply.is_file() and parts_path.is_file(): break
    else:
        raise ValueError(f'{parent.name}: no complete mesh and labels')
    info = json.loads(parts_path.read_text(encoding='utf-8'))
    source = Path(report['source'])
    mesh_hash, raw_hash = file_hash(ply), file_hash(source)
    if info['mesh']['sha256'] != mesh_hash or report['fingerprint']['source_sha256'] != raw_hash:
        raise ValueError('Source/mesh does not match labels and report.')
    mesh = o3d.io.read_triangle_mesh(str(ply))
    vertices = np.asarray(mesh.vertices)
    cloud = o3d.io.read_point_cloud(str(source)).remove_non_finite_points()
    if report['fingerprint']['unit'] == 'm': cloud.scale(1000, center=np.zeros(3))
    raw = np.asarray(cloud.points)
    labels = np.full(len(vertices), 'unknown', dtype=object)
    for name, part in info['parts'].items(): labels[part['vertex_indices']] = name
    nearest = cKDTree(vertices).query(raw, workers=4)[1]
    out = ROOT/'outputs/experiments/hand_branches'/parent.name
    out.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (1400, 1520), 'white')
    draw = ImageDraw.Draw(canvas)
    records = []
    for side, name in enumerate(('left_hand', 'right_hand')):
        ids, arm_class, wrist, axes = hand_frame(vertices, info['parts'], name)
        local_vertices = (vertices-wrist) @ axes
        low = local_vertices[ids].min(0)-[12, 30, 15]
        high = local_vertices[ids].max(0)+[12, 12, 15]
        low[1] = -30
        local = (raw-wrist) @ axes
        selected = (np.all((local >= low) & (local <= high), axis=1)
            & (cKDTree(vertices[ids]).query(raw, workers=4)[0] < 22)
            & np.isin(labels[nearest], [name, arm_class, 'unknown']))
        points = local[selected]
        if len(points) < 100:
            records.append({'part': name, 'status': 'insufficient_points'}); continue
        branches, palm, diagnostics = assess(points)
        line_points, line_edges, line_colors = [], [], []
        for branch in branches:
            path = np.array(branch['local_path_mm']) @ axes.T+wrist
            branch['world_path_mm'] = path.tolist()
            n = len(line_points)
            line_points.extend(path.tolist())
            line_edges.extend([[n+i, n+i+1] for i in range(len(path)-1)])
            color = [0., .65, .2] if branch['status'] == 'supported_candidate' else [1., .5, 0.]
            line_colors.extend([color]*(len(path)-1))
        lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(line_points).reshape(-1, 3)),
                                     o3d.utility.Vector2iVector(np.array(line_edges, int).reshape(-1, 2)))
        lines.colors = o3d.utility.Vector3dVector(np.array(line_colors).reshape(-1, 3))
        lines_path = out/f'{name}_centerlines.ply'
        if line_edges:
            if not o3d.io.write_line_set(str(lines_path), lines):
                raise OSError('Cannot write centerlines.')
            restored = o3d.io.read_line_set(str(lines_path))
            if len(restored.lines) != len(line_edges) or not np.allclose(restored.points, line_points):
                raise OSError('Centerline PLY did not round-trip.')
        else:
            lines_path.unlink(missing_ok=True)
        old_path = ROOT/'outputs/experiments/hand_density'/parent.name/f'{name}_projected_skeleton.ply'
        old = np.asarray(o3d.io.read_point_cloud(str(old_path)).points) if old_path.is_file() else np.empty((0, 3))
        center = wrist+axes @ ((low+high)/2)
        span = max(high-low)*1.12
        for angle in (0, 1):
            row = side*2+angle
            right, up = (axes[:, 0], axes[:, 1]) if angle == 0 else (axes[:, 2], axes[:, 1])
            for column, title in enumerate(('Original points', 'Previous skeleton', 'Palm + candidates', 'Supported candidates')):
                picture = point_panel(raw[selected], center, right, up, span)
                overlay = ImageDraw.Draw(picture)
                def uv(world):
                    return np.c_[((world-center) @ right/span+.5)*349,
                                  (1-((world-center) @ up/span+.5))*349]
                if column == 1:
                    for x, y in uv(old): overlay.ellipse((x-1, y-1, x+1, y+1), fill='#e83838')
                if column == 2 and palm is not None:
                    pxy = np.argwhere(palm['mask'])[::4]*palm['voxel_mm']+palm['origin']
                    world = np.c_[pxy, np.full(len(pxy), np.median(points[:, 2]))] @ axes.T+wrist
                    for x, y in uv(world): overlay.ellipse((x-1, y-1, x+1, y+1), fill='#4d9fe6')
                if column >= 2:
                    for i, branch in enumerate(branches):
                        supported = branch['status'] == 'supported_candidate'
                        if column == 3 and not supported: continue
                        coords = uv(np.array(branch['world_path_mm']))
                        color = '#00ad35' if supported else '#f09b16'
                        overlay.line([tuple(p) for p in coords], fill=color, width=3)
                        x, y = coords[-1]; overlay.ellipse((x-3, y-3, x+3, y+3), fill=color)
                        overlay.text((x+3, y), str(i+1), fill='#a00000')
                canvas.paste(picture, (column*350, row*380+25))
                draw.text((column*350+4, row*380+5), f'{name} | {title}', fill='black')
        record = {'part': name, 'raw_points': len(points), 'wrist_mm': wrist.tolist(),
                  'frame_axes': axes.tolist(), 'branches': branches, 'diagnostics': diagnostics,
                  'supported_candidates': sum(b['status']=='supported_candidate' for b in branches),
                  'palm_center_local_xy_mm': palm['palm_center_xy'].tolist() if palm else None,
                  'palm_radius_mm': palm['palm_radius_mm'] if palm else None}
        records.append(record)
        print(parent.name, name, 'candidates', len(branches), 'supported', record['supported_candidates'],
              'stability', [round(b['stability'], 2) for b in branches], flush=True)
    canvas.save(out/'comparison.jpg', quality=94)
    if file_hash(ply) != mesh_hash or file_hash(source) != raw_hash:
        raise ValueError('Original model or raw input changed during trial.')
    result = {'source_mesh': str(ply), 'source_sha256': mesh_hash, 'source_cloud': str(source),
              'source_cloud_sha256': raw_hash, 'parts': records, 'accepted_for_full_body': False,
              'scope': 'Geometric centerline candidates, not anatomical joints or recovered missing fingers.',
              'confidence_note': 'Heuristic evidence scores, not calibrated probabilities. Scale stability is not accuracy.',
              'elapsed_seconds': round(time.perf_counter()-started, 2)}
    (out/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(parent.name, 'seconds', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        for scan in args.scan: trial(scan)
