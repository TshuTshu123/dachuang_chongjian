"""对照举手与垂手的原始观测，并测试手部局部重建。

项目根目录运行 python tools/hand_pose_trial.py --scan 0 1 2 3。
读取原点云及匹配的 Meta 标签，以手腕到手指方向统一观察坐标。
输出 outputs/experiments/hand_pose/ 的对比图与报告。
仅在外扩明显收敛时保留 balanced 局部预览，腕口未拼接，不能作为完整模型。
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from body_parts import file_hash
from surface_evaluation import distances
from surface_detail import denoise_local_planes, fit_local_planes
from tools.surface_trials import panel


def hand_frame(vertices, parts, name):
    ids = np.asarray(parts[name]['vertex_indices'], dtype=int)
    arm_class = name.replace('hand', 'lower_arm')
    arm = np.asarray(parts[arm_class]['vertex_indices'], dtype=int)
    if len(arm) < 30:
        arm_class = 'upper_clothing'
        arm = np.asarray(parts[arm_class]['vertex_indices'], dtype=int)
    if min(len(ids), len(arm)) < 30:
        raise ValueError('Insufficient hand or forearm labels.')
    nearest_arm = cKDTree(vertices[arm]).query(vertices[ids])[0]
    wrist = np.median(vertices[ids[nearest_arm <= np.quantile(nearest_arm, .08)+3]], axis=0)
    _, axes = np.linalg.eigh(np.cov(vertices[ids].T))
    normal = axes[:, 0]
    forward = vertices[ids].mean(0) - wrist
    forward -= (forward @ normal)*normal
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, normal)
    return ids, arm_class, wrist, np.column_stack((right, forward, normal))


def point_panel(points, center, right, up, span, size=350):
    xy = np.c_[(points-center) @ right, (points-center) @ up]
    uv = np.rint((xy/span+.5)*(size-1)).astype(int)
    image = Image.new('RGB', (size, size), '#f0f0f0')
    draw = ImageDraw.Draw(image)
    for x, y in uv:
        if 0 <= x < size and 0 <= y < size:
            draw.ellipse((x-1, size-y-2, x+1, size-y), fill='#303b4a')
    return image


def local_mesh(cloud, spacing, method):
    cloud = o3d.geometry.PointCloud(cloud)
    radius = float(np.clip(spacing*2.5, 3., 7.))
    if method == 'balanced':
        guide = np.asarray(cloud.normals).copy()
        cleaned, _ = denoise_local_planes(np.asarray(cloud.points)*.001, 24, .006, 4, 2, max_move=.0007)
        normal, _, _, _, _ = fit_local_planes(cleaned, 24, .006, 4)
        normal[np.einsum('ij,ij->i', normal, guide)<0] *= -1
        cloud.points = o3d.utility.Vector3dVector(cleaned*1000)
        cloud.normals = o3d.utility.Vector3dVector(normal)
        cloud.orient_normals_consistent_tangent_plane(20)
        radius = 6.
    elif method == 'local':
        guide = np.asarray(cloud.normals).copy()
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=24))
        normals = np.asarray(cloud.normals)
        normals[np.einsum('ij,ij->i', normals, guide)<0] *= -1
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=6 if method == 'balanced' else 8, scale=1.05, linear_fit=True, n_threads=4)
    if method == 'balanced':
        mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
    points = np.asarray(cloud.points)
    vertices = np.asarray(mesh.vertices)
    unsupported = cKDTree(points).query(vertices)[0] > max(4., 2.5*spacing)
    outside = np.any((vertices < points.min(0)-2) | (vertices > points.max(0)+2), axis=1)
    mesh.remove_vertices_by_mask(unsupported | outside)
    mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
    nearest = cKDTree(points).query(np.asarray(mesh.vertices))[1]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[nearest])
    mesh.compute_vertex_normals()
    return mesh, radius


def trial(scan):
    started = time.perf_counter()
    name = f'export_{scan:05d}'
    for path in sorted((ROOT/'outputs/models'/name).glob('*/report.json'), reverse=True):
        report = json.loads(path.read_text(encoding='utf-8'))
        ply = path.parent/report['artifacts']['ply']['name']
        if ply.is_file() and (path.parent/'body_parts.json').is_file():
            break
    else:
        raise ValueError(f'{name}: completed mesh with labels not found.')
    info = json.loads((path.parent/'body_parts.json').read_text(encoding='utf-8'))
    digest = file_hash(ply)
    if info['mesh']['sha256'] != digest:
        raise ValueError('Body labels do not match mesh.')
    mesh = o3d.io.read_triangle_mesh(str(ply))
    vertices = np.asarray(mesh.vertices)
    source_cloud = Path(report['source'])
    cloud_hash = file_hash(source_cloud)
    if cloud_hash != report['fingerprint']['source_sha256']:
        raise ValueError('Original point cloud differs from the saved reconstruction input.')
    cloud = o3d.io.read_point_cloud(str(source_cloud)).remove_non_finite_points()
    if report['fingerprint']['unit'] == 'm':
        cloud.scale(1000, center=np.zeros(3))
    raw = np.asarray(cloud.points)
    if not cloud.has_normals():
        raise ValueError('This trial needs the original acquisition normals.')
    labels = np.full(len(vertices), 'unknown', dtype=object)
    for part, data in info['parts'].items():
        labels[data['vertex_indices']] = part
    raw_distance, nearest = cKDTree(vertices).query(raw, workers=4)
    raw_labels = labels[nearest]
    out = ROOT/'outputs/experiments/hand_pose'/name
    out.mkdir(parents=True, exist_ok=True)
    records = []
    canvas = Image.new('RGB', (1750, 4*380), 'white')
    draw = ImageDraw.Draw(canvas)
    for side, part in enumerate(('left_hand', 'right_hand')):
        ids, arm_class, wrist, axes = hand_frame(vertices, info['parts'], part)
        frame = (vertices-wrist) @ axes
        palm_length = float(np.quantile(frame[ids, 1], .98))
        low = frame[ids].min(0)-[12, 30, 15]
        high = frame[ids].max(0)+[12, 12, 15]
        low[1] = -30
        local = (raw-wrist) @ axes
        near_hand = cKDTree(vertices[ids]).query(raw, workers=4)[0] < 22
        allowed = np.isin(raw_labels, (part, arm_class, 'unknown'))
        observed = np.all((local >= low) & (local <= high), axis=1) & near_hand & allowed
        raw_ids = np.flatnonzero(observed)
        if len(raw_ids) < 100:
            records.append({'part': part, 'status': 'insufficient_raw_points', 'raw_points': len(raw_ids)})
            continue
        hand_cloud = cloud.select_by_index(raw_ids).remove_duplicated_points()
        hand_raw = np.asarray(hand_cloud.points)
        spacing = float(np.median(cKDTree(hand_raw).query(hand_raw, k=2)[0][:, 1]))
        enclosed = np.all((frame >= low) & (frame <= high), axis=1)
        permitted = np.isin(labels, (part, arm_class, 'unknown'))
        baseline = copy.deepcopy(mesh)
        baseline.remove_vertices_by_mask(~(enclosed & permitted))
        baseline.compute_vertex_normals()
        baseline_support = cKDTree(raw).query(vertices[ids], workers=4)[0]
        baseline_fit = float(np.quantile(distances(mesh, hand_raw), .95))
        candidates = []
        for method in ('original', 'local', 'balanced'):
            candidate, radius = local_mesh(hand_cloud, spacing, method)
            support = cKDTree(hand_raw).query(np.asarray(candidate.vertices))[0]
            candidates.append(candidate)
            candidate_file = out / f'{part}_{method}_normals.ply'
            fit = float(np.quantile(distances(candidate, hand_raw), .95))
            keep_preview = (method == 'balanced'
                            and np.quantile(baseline_support, .95) > np.quantile(support, .95)+2
                            and fit <= baseline_fit+.5)
            if keep_preview:
                if not o3d.io.write_triangle_mesh(str(candidate_file), candidate):
                    raise OSError(f'Cannot export {candidate_file}')
            else:
                candidate_file.unlink(missing_ok=True)
            records.append({'part': part, 'method': method+'_normals',
                            'raw_points': len(hand_raw), 'spacing_median_mm': spacing,
                            'normal_radius_mm': radius, 'raw_to_mesh_p95_mm': fit,
                            'mesh_to_raw_p95_mm': float(np.quantile(support, .95)),
                            'boundary_edges': len(candidate.get_non_manifold_edges(allow_boundary_edges=False)),
                            'candidate': str(candidate_file) if keep_preview else None,
                            'preview_only': True, 'accepted_for_full_body': False})
        unsupported = baseline_support
        other = np.flatnonzero(~np.isin(labels, ('unknown', part, arm_class)))
        contact = cKDTree(vertices[other]).query(vertices[ids])[0]
        records.append({'part': part, 'method': 'baseline', 'raw_points': len(hand_raw),
                        'spacing_median_mm': spacing,
                        'mesh_to_raw_p95_mm': float(np.quantile(unsupported, .95)),
                        'raw_to_mesh_p95_mm': float(np.quantile(distances(mesh, hand_raw), .95)),
                        'vertices_over_5mm_from_raw_fraction': float((unsupported>5).mean()),
                        'other_body_distance_q05_mm': float(np.quantile(contact, .05)),
                        'hand_label_vertices': len(ids), 'length_mm': palm_length,
                        'wrist_mm': wrist.tolist(), 'frame_axes': axes.tolist()})
        center = wrist + axes @ ((low+high)/2)
        span = max(high-low)*1.12
        for angle in (0, 1):
            row = side*2+angle
            right, up = (axes[:, 0], axes[:, 1]) if angle == 0 else (axes[:, 2], axes[:, 1])
            picture = point_panel(hand_raw, center, right, up, span)
            canvas.paste(picture, (0, row*380+25))
            draw.text((5, row*380+5), f'{part} | original points | view {angle}', fill='black')
            for column, (model, title) in enumerate(zip([baseline, *candidates],
                                            ('Current mesh', 'Local: acquisition normals', 'Local: small normals', 'Local: balanced')), 1):
                canvas.paste(panel(model, center, right, up, span, 350), (column*350, row*380+25))
                draw.text((column*350+5, row*380+5), title, fill='black')
        print(name, part, 'raw', len(hand_raw), 'spacing', round(spacing, 2), flush=True)
    canvas.save(out/'comparison.jpg', quality=95)
    assert file_hash(ply) == digest
    assert file_hash(source_cloud) == cloud_hash
    result = {'source': str(ply), 'source_sha256': digest, 'source_cloud_sha256': cloud_hash,
              'scan': scan, 'parts': records,
              'scope': 'Independent local reconstructions, open at wrist; not stitched into full body.',
              'elapsed_seconds': round(time.perf_counter()-started, 2)}
    (out/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(name, 'seconds', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, nargs='+', default=[0, 1, 2, 3])
    for scan in parser.parse_args().scan:
        trial(scan)
