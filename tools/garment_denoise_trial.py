"""比较衣裤点云在 Poisson 重建前局部去噪的效果。

在项目根目录运行 python tools/garment_denoise_trial.py --scan 1 2。
--radius-mm 8 可比较较宽的拟合邻域，默认 5 mm。
用同一批训练点和重建参数，留出部分衣裤点做对照。只保留灰模对比和报告。
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from multiview_geometry_trial import ROOT, reconstruct
from surface_detail import fit_local_planes
from surface_evaluation import distances, metrics
from surface_quality import regions
from tools.surface_trials import panel
from body_parts import file_hash


GARMENTS = ('upper_clothing', 'lower_clothing')
RADIUS_MM = 5.0
MAX_SHIFT_MM = 1.2


def denoise(points, train, labels, weight, radius):
    shift = np.zeros_like(points)
    support = {}
    for index, name in enumerate(GARMENTS, 1):
        ids = np.flatnonzero(train & (labels == index))
        if len(ids) < 12:
            raise ValueError(f'Insufficient training points for {name}.')
        normal, offset, _, flat, counts = fit_local_planes(points[ids]*.001, 32, radius*.001, 4)
        eligible = flat & (counts >= 10) & (weight[ids] > .1)
        step = np.clip(.85*offset*1000, -MAX_SHIFT_MM, MAX_SHIFT_MM) * weight[ids]
        step[~eligible] = 0
        shift[ids] = step[:, None] * normal
        support[name] = {
            'eligible_points': int(eligible.sum()),
            'moved_points_over_0_1mm': int((np.abs(step) > .1).sum()),
            'movement_max_mm': float(np.abs(step).max()),
        }
    return shift, support


def comparison(before, after, ref, labels, out):
    up = np.array([1., 0., 0.])
    right = np.array([0., 1., 0.])
    size = 460
    views = [(1, right, 'Shirt front'), (1, -right, 'Shirt back'),
             (2, right, 'Trousers front'), (2, -right, 'Trousers back')]
    canvas = Image.new('RGB', (2*size, len(views)*(size+24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (part, direction, name) in enumerate(views):
        points = ref[labels == part]
        center = (points.min(axis=0)+points.max(axis=0))/2
        span = np.ptp(points, axis=0).max()*1.04
        top = row*(size+24)
        for col, (mesh, text) in enumerate(((before, 'Baseline'), (after, 'Point denoise'))):
            canvas.paste(panel(mesh, center, direction, up, span, size), (col*size, top+24))
            draw.text((col*size+8, top+6), f'{text} | {name}', fill='black')
    canvas.save(out/'comparison.jpg', quality=95)


def trial(scan, radius):
    name = f'export_{scan:05d}'
    report_path = next(path for path in sorted((ROOT/'outputs/models'/name).glob('*/report.json'), reverse=True)
                       if (path.parent/'body_parts.json').is_file())
    report = json.loads(report_path.read_text(encoding='utf-8'))
    parts = json.loads((report_path.parent/'body_parts.json').read_text(encoding='utf-8'))
    source = Path(report['source'])
    mesh_path = report_path.parent/report['artifacts']['ply']['name']
    mesh_hash = file_hash(mesh_path)
    if mesh_hash != parts['mesh']['sha256'] or file_hash(source) != report['fingerprint']['source_sha256']:
        raise ValueError('The input or mesh differs from the saved labels and report.')
    reference = o3d.io.read_triangle_mesh(str(mesh_path))
    ref_xyz = np.asarray(reference.vertices)
    cloud = o3d.io.read_point_cloud(str(source)).remove_non_finite_points()
    if report['fingerprint']['unit'] == 'm':
        cloud.scale(1000, center=(0., 0., 0.))
    points = np.asarray(cloud.points)
    tree = cKDTree(ref_xyz)
    distance, nearest = tree.query(points, workers=4)
    ref_labels = np.zeros(len(ref_xyz), dtype=np.int8)
    for index, region in enumerate(GARMENTS, 1):
        ref_labels[parts['parts'][region]['vertex_indices']] = index
    labels = np.where(distance <= 5., ref_labels[nearest], 0)
    garment_parts = {name: parts['parts'][name] for name in GARMENTS}
    garment_parts['face_neck'] = {'vertex_indices': []}
    influence = regions(reference, garment_parts)[0][nearest].copy()
    for index in range(1, len(GARMENTS)+1):
        ids = labels == index
        boundary = cKDTree(ref_xyz[ref_labels != index]).query(points[ids], workers=4)[0]
        influence[ids] *= np.clip((boundary - radius) / radius, 0, 1)

    rng = np.random.default_rng(17)
    held_out = (labels > 0) & (rng.random(len(points)) < .2)
    train = ~held_out
    shift, support = denoise(points, train, labels, influence, radius)
    before = cloud.select_by_index(np.flatnonzero(train))
    candidate = o3d.geometry.PointCloud(before)
    candidate.points = o3d.utility.Vector3dVector((points+shift)[train])
    out = ROOT/'outputs/experiments/garment_point_denoise'/name
    if radius != RADIUS_MM:
        out /= f'radius_{radius:g}mm'
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='garment-', dir=out) as temporary:
        work = Path(temporary)
        settings = report['base_report']['settings']
        baseline, base_qa = reconstruct(work/'baseline.ply', before, work/'base', settings)
        improved, improved_qa = reconstruct(work/'candidate.ply', candidate, work/'new', settings)

    result = {'source': str(source), 'reference': str(mesh_path), 'reference_sha256': mesh_hash,
              'radius_mm': radius, 'point_shift_cap_mm': MAX_SHIFT_MM,
              'holdout': 'Random 20% within clothing, excluded from both runs and local fitting.',
              'regions': {}}
    meshes = (baseline, improved)
    remapped = [ref_labels[tree.query(np.asarray(mesh.vertices), workers=4)[1]] for mesh in meshes]
    for index, region in enumerate(GARMENTS, 1):
        samples = points[held_out & (labels == index)]
        values = [metrics(mesh, points, samples, None, mapping == index, mapping == index)
                  for mesh, mapping in zip(meshes, remapped)]
        result['regions'][region] = {**support[region], 'held_out_points': len(samples),
                                     'baseline': values[0], 'candidate': values[1]}
    result['protected_geometry'] = {}
    for region in ('face_neck', 'left_hand', 'right_hand', 'left_shoe', 'right_shoe'):
        samples = ref_xyz[parts['parts'][region]['vertex_indices']]
        if len(samples):
            delta = distances(improved, samples)-distances(baseline, samples)
            result['protected_geometry'][region] = {'samples': len(samples),
                'absolute_change_in_reference_distance_p95_mm': float(np.quantile(np.abs(delta), .95))}
    keys = ('triangles', 'boundary_edges', 'edges_with_more_than_two_faces', 'orientation_succeeded')
    result['topology'] = {name: {key: qa[key] for key in keys} for name, qa in
                          (('baseline', base_qa), ('candidate', improved_qa))}
    comparison(baseline, improved, ref_xyz, ref_labels, out)
    (out/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, nargs='+', required=True)
    parser.add_argument('--radius-mm', type=float, default=RADIUS_MM)
    args = parser.parse_args()
    if not np.isfinite(args.radius_mm) or args.radius_mm <= 0:
        parser.error('--radius-mm must be positive and finite')
    for scan in args.scan:
        trial(scan, args.radius_mm)
