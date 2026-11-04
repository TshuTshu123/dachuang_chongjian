"""检查逐路原始导出、按标定合并，并用其他视角的有效颜色补缺色黑点。
根目录执行：python tools/raw_capture.py "data_raw/raw(2)/raw"
--frame 00000：选择共同帧；--out 目录：输出位置；--color-radius-mm 8：补色最大距离。
--mesh 网格.ply：在同一网格上生成补色前后对比图（不修改网格）。
只写 outputs，不修改原始数据。输出两份同坐标点云、检查报告和照片总览。
当前读取 mm 制 PLY；相机投影约定会逐路核验，核验失败的路不参与补色。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def primary_projection(points, camera):
    # Export convention, checked against the existing per-point RGB below.
    rotation = np.asarray(camera['rotation'], dtype=float)[:3, :3]
    local = (points - np.asarray(camera['translation'])) @ rotation
    z = local[:, 2]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        x, y = -local[:, 0] / z, -local[:, 1] / z
        p1, p2, k1, k2, k3 = camera['distortion']
        r2 = x*x + y*y
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        xx = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        yy = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        uv = np.column_stack((camera['F'][0]*xx + camera['C'][0],
                              camera['F'][1]*yy + camera['C'][1]))
    valid = ((z > 0) & np.isfinite(uv).all(axis=1) &
             (uv[:, 0] >= 2) & (uv[:, 0] < camera['width'] - 3) &
             (uv[:, 1] >= 2) & (uv[:, 1] < camera['height'] - 3))
    return uv, valid


def make_overview(stream_dirs, frame, destination):
    columns = 6
    rows = 2 * ((len(stream_dirs) + columns - 1) // columns)
    canvas = Image.new('RGB', (columns*300, rows*270), '#222222')
    draw = ImageDraw.Draw(canvas)
    for i, directory in enumerate(stream_dirs):
        for side in (1, 2):
            x, y = (i % columns)*300, ((i // columns)*2 + side-1)*270
            with Image.open(directory / f'{frame}_RGB_{side:02d}.png') as image:
                image.thumbnail((294, 240))
                canvas.paste(image, (x + (300-image.width)//2, y+25))
            draw.text((x+8, y+6), f'stream {directory.name} RGB {side:02d}', fill='white')
    canvas.save(destination, quality=90)


def compare_mesh(mesh_path, out):
    """Compare vertex colors on identical geometry, with fixed cameras and lighting."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError('Comparison requires a triangle mesh')
    original = o3d.io.read_point_cloud(str(out/'aligned_original.ply'))
    repaired = o3d.io.read_point_cloud(str(out/'aligned_color_repaired.ply'))
    points = np.asarray(original.points)
    vertices = np.asarray(mesh.vertices)
    distance, nearest = cKDTree(points).query(vertices, workers=4)
    if np.median(distance) > 5 or np.percentile(distance, 95) > 12:
        raise ValueError('Mesh does not closely match these observations in millimeters')
    before = np.asarray(original.colors)[nearest]
    after = np.asarray(repaired.colors)[nearest]
    faces = np.asarray(mesh.triangles)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    up = np.eye(3)[np.argmax(np.ptp(vertices, axis=0))]
    # Preview axes only; no model transform or semantic detector is applied.
    axes = [i for i in range(3) if up[i] == 0]
    forward = -np.eye(3)[axes[-1]]
    side = -np.eye(3)[axes[0]]
    center = (vertices.min(axis=0)+vertices.max(axis=0))/2
    span = np.ptp(vertices, axis=0).max()*1.05
    canvas = Image.new('RGB', (1000, 1480), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, outward in enumerate((forward, side)):
        right = np.cross(up, outward)
        width, height = 500, 700
        xx, yy = np.meshgrid(np.linspace(-span*width/height/2, span*width/height/2, width),
                             np.linspace(span/2, -span/2, height))
        origins = center + xx[..., None]*right + yy[..., None]*up + outward*3000
        rays = np.c_[origins.reshape(-1, 3), np.tile(-outward, (width*height, 1))].astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
        ok = np.isfinite(hit['t_hit'].numpy())
        triangles = faces[hit['primitive_ids'].numpy()[ok]]
        uv = hit['primitive_uvs'].numpy()[ok]
        weights = np.c_[1-uv.sum(axis=1), uv]
        normal = hit['primitive_normals'].numpy()[ok]
        light = outward + .3*up + .2*right
        light /= np.linalg.norm(light)
        shade = .55+.45*np.abs(normal @ light)
        for column, colors in enumerate((before, after)):
            pixels = np.full((width*height, 3), .9)
            pixels[ok] = np.clip(np.einsum('ij,ijk->ik', weights, colors[triangles])*shade[:, None], 0, 1)
            image = Image.fromarray(np.rint(pixels.reshape(height, width, 3)*255).astype(np.uint8))
            y = row*740
            canvas.paste(image, (column*width, y+30))
            draw.text((column*width+10, y+8), ('Original colors' if column == 0 else 'Missing colors repaired') +
                      (' - front' if row == 0 else ' - side'), fill='black')
    canvas.save(out/'color_comparison.jpg', quality=94)
    stats = {'mesh': str(mesh_path.resolve()), 'geometry_identical_in_both_panels': True,
             'vertices': len(vertices), 'triangles': len(faces),
             'nearly_black_vertices_before': int((before.max(axis=1) <= 1/255).sum()),
             'nearly_black_vertices_after': int((after.max(axis=1) <= 1/255).sum()),
             'changed_vertex_colors': int(np.any(before != after, axis=1).sum()),
             'mesh_to_raw_distance_mm_p50_p95': np.percentile(distance, [50, 95]).tolist(),
             'note': 'Color comparison only; darkness of real material and overall exposure are not optimized.'}
    (out/'color_comparison.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2), flush=True)


def inspect_capture(folder, frame, out, radius):
    directories = sorted((p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()),
                         key=lambda p: int(p.name))
    if not directories:
        raise ValueError('No numbered stream directories')
    common = set.intersection(*({p.stem for p in d.glob('*.ply') if p.stem.isdigit()}
                               for d in directories))
    if not common:
        raise ValueError('No frame has a PLY in every stream')
    frame = frame or sorted(common)[0]
    if frame not in common:
        raise ValueError(f'Incomplete frame {frame}; common frames: {sorted(common)}')
    out.mkdir(parents=True, exist_ok=True)
    records, positions, normals, colors, valid_colors, missing_colors = [], [], [], [], [], []
    for directory in directories:
        paths = [directory/f'{frame}.ply', directory/f'{frame}_CameraParams_Primary.json',
                 directory/f'{frame}_CameraParams_Secondary.json',
                 directory/f'{frame}_transform.txt', directory/f'{frame}_PhotogrammetryInfo.txt',
                 directory/f'{frame}_RGB_01.png', directory/f'{frame}_RGB_02.png',
                 directory/'CylinderCropParams.cfg']
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise ValueError(f'Missing required files: {missing}')
        if not re.search(r'Units:\s*mm\b', paths[-1].read_text()):
            raise ValueError('Expected explicit millimeter units in CylinderCropParams.cfg')
        matrix = np.loadtxt(directory/f'{frame}_transform.txt')
        rotation = matrix[:3, :3]
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
            abs(np.linalg.det(rotation)-1) > 1e-4):
            raise ValueError(f'Invalid rigid transform in stream {directory.name}')
        cloud = o3d.io.read_point_cloud(str(paths[0]))
        if not cloud.has_points() or not cloud.has_colors() or not cloud.has_normals():
            raise ValueError(f'PLY lacks points, colors or normals: {paths[0]}')
        points = np.asarray(cloud.points).copy()
        rgb = np.asarray(cloud.colors).copy()
        normal = np.asarray(cloud.normals).copy()
        if not np.isfinite(points).all() or not np.isfinite(normal).all():
            raise ValueError(f'Non-finite point data: {paths[0]}')
        length = np.linalg.norm(normal, axis=1)
        normal /= np.maximum(length[:, None], 1e-12)
        normal[np.einsum('ij,ij->i', normal, -points) < 0] *= -1
        camera = json.loads(paths[1].read_text())['color']
        uv, inside = primary_projection(points, camera)
        with Image.open(directory/f'{frame}_RGB_01.png') as image:
            if image.size != (camera['width'], camera['height']):
                raise ValueError('Image size and calibration disagree')
            pixels = np.asarray(image.convert('RGB'))
        with Image.open(directory/f'{frame}_RGB_02.png') as image:
            image.verify()
        ids = np.flatnonzero(inside)[::max(1, int(inside.sum())//3000)]
        if not len(ids):
            raise ValueError('No points project inside the primary image')
        xy = np.rint(uv[ids]).astype(int)
        color_error = float(np.median(np.abs(pixels[xy[:, 1], xy[:, 0]]/255-rgb[ids]).mean(axis=1)))
        projection_ok = bool(inside.mean() > .6 and color_error < .06)
        nearly_black = rgb.max(axis=1) <= 1/255
        global_points = transform_points(points, matrix)
        global_normals = normal @ rotation.T
        positions.append(global_points)
        normals.append(global_normals)
        colors.append(rgb)
        # An in-frame black material remains a legitimate color; do not threshold donors by brightness.
        valid_colors.append(inside & projection_ok)
        missing_colors.append(~inside & nearly_black & projection_ok)
        records.append({'stream': directory.name, 'points': len(points),
                        'primary_projection_valid_fraction': float(inside.mean()),
                        'projection_color_median_error_0_1': color_error,
                        'projection_check_passed': projection_ok,
                        'nearly_black_points': int(nearly_black.sum()),
                        'out_of_image_black_points': int((~inside & nearly_black).sum()),
                        'transform': matrix.tolist(),
                        'files': [{'path': str(p.relative_to(folder)), 'bytes': p.stat().st_size,
                                   'sha256': sha256(p)} for p in paths]})
        print(f"stream {directory.name}: {len(points)} points, projection {inside.mean():.1%}, "
              f"color error {color_error:.4f}, passed={projection_ok}", flush=True)

    points = np.concatenate(positions)
    normal = np.concatenate(normals)
    rgb = np.concatenate(colors)
    reliable = np.concatenate(valid_colors)
    # Repair only missing near-zero colors outside the calibrated image, never in-frame black material.
    targets = np.flatnonzero(np.concatenate(missing_colors))
    donors = np.flatnonzero(reliable)
    repaired = rgb.copy()
    changed = np.zeros(len(points), dtype=bool)
    if len(donors) and len(targets):
        tree = cKDTree(points[donors])
        distance, index = tree.query(points[targets], k=min(16, len(donors)),
                                     distance_upper_bound=radius, workers=4)
        if distance.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        found = np.isfinite(distance)
        safe_index = donors[np.minimum(index, len(donors)-1)]
        agreement = np.einsum('ijk,ik->ij', normal[safe_index], normal[targets])
        eligible = found & (agreement > .7)
        cost = np.where(eligible, distance, np.inf)
        best = cost.argmin(axis=1)
        good = np.isfinite(cost[np.arange(len(targets)), best])
        selected = targets[good]
        repaired[selected] = rgb[safe_index[np.arange(len(targets))[good], best[good]]]
        changed[selected] = True
    baseline = o3d.geometry.PointCloud()
    baseline.points = o3d.utility.Vector3dVector(points)
    baseline.normals = o3d.utility.Vector3dVector(normal)
    baseline.colors = o3d.utility.Vector3dVector(rgb)
    if not o3d.io.write_point_cloud(str(out/'aligned_original.ply'), baseline):
        raise IOError('Failed to write aligned point cloud')
    baseline.colors = o3d.utility.Vector3dVector(repaired)
    if not o3d.io.write_point_cloud(str(out/'aligned_color_repaired.ply'), baseline):
        raise IOError('Failed to write repaired point cloud')
    # Verify exact geometry/normals and that valid source colors were untouched after serialization.
    saved = o3d.io.read_point_cloud(str(out/'aligned_color_repaired.ply'))
    assert np.array_equal(np.asarray(saved.points), points)
    assert np.array_equal(np.asarray(saved.normals), normal)
    assert np.max(np.abs(np.asarray(saved.colors)[reliable]-rgb[reliable])) < .5/255 + 1e-9
    make_overview(directories, frame, out/'images_overview.jpg')
    starts = np.r_[0, np.cumsum([len(v) for v in positions])]
    for i, record in enumerate(records):
        indices = np.arange(starts[i], starts[i+1])
        sample = indices[::max(1, len(indices)//1000)]
        other = np.ones(len(points), bool)
        other[indices] = False
        nearest = cKDTree(points[other]).query(points[sample], workers=4)[0]
        record['cross_view_nearest_mm_p50_p90'] = np.percentile(nearest, [50, 90]).tolist()
        record['repaired_points'] = int(changed[indices].sum())
    all_files = [p for p in folder.rglob('*') if p.is_file()]
    report = {'source': str(folder), 'frame': frame, 'complete_frames': sorted(common),
              'unit': 'mm', 'stream_count': len(directories), 'total_points': len(points),
              'extensions': dict(Counter(p.suffix.lower() for p in all_files)),
              'projection_convention': 'camera_to_point_frame inverse, negative x/y before pinhole; distortion p1 p2 k1 k2 k3, per-stream photometric check',
              'nearly_black_points': int((rgb.max(axis=1) <= 1/255).sum()),
              'repair_candidates': len(targets), 'repaired_points': int(changed.sum()),
              'repair_max_neighbor_distance_mm': radius, 'repair_min_normal_cosine': .7,
              'geometry_changed': False, 'streams': records,
              'notes': ['Frame filenames do not prove timestamp synchronization or matching to an existing merged PLY.',
                        'Camera convention validation is photometric, not an independent ground-truth calibration.',
                        'Color repair copies nearby valid observations; it does not create new geometric detail.',
                        'Secondary RGB images are retained for future calibrated texturing, not used by this color repair.']}
    (out/'raw_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('stream_count', 'total_points', 'nearly_black_points',
                                             'repair_candidates', 'repaired_points')}, indent=2), flush=True)
    print('Output:', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--frame', help='Exported frame prefix; default: first common frame')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--mesh', type=Path, help='Generate same-geometry color comparison')
    parser.add_argument('--color-radius-mm', type=float, default=8)
    args = parser.parse_args()
    if not np.isfinite(args.color_radius_mm) or args.color_radius_mm <= 0:
        parser.error('--color-radius-mm must be positive and finite')
    folder = args.capture.resolve()
    destination = args.out or ROOT/'outputs/capture_inputs'/f'{folder.parent.name.replace("(", "").replace(")", "")}_frame{args.frame or "00000"}'
    if destination.resolve().is_relative_to(folder):
        parser.error('Output must be outside the original input directory')
    inspect_capture(folder, args.frame, destination.resolve(), args.color_radius_mm)
    if args.mesh:
        compare_mesh(args.mesh, destination.resolve())
